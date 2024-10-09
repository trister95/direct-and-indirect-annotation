import torch
import higher
import evaluate
from functools import partial
from .utils import ugly_log 
import numpy as np
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

def filter_none_labels(dataset):
    """
    Filters out examples with None labels from the dataset.
    """
    def has_valid_labels(example):
        # Check if the labels are not None
        return example['labels'] is not None
    
    # Filter the dataset
    return dataset.filter(has_valid_labels)

def compute_metrics(p, label_array, log_file):
    predictions, labels = p
    # Check if predictions or labels are None
    if predictions is None or labels is None:
        ugly_log(log_file, "Predictions or labels are None in compute_metrics")
        return {}
    
    try:
        predictions = np.argmax(predictions, axis=2)
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()

        valid_mask = labels_flat != -100
        true_predictions = label_array[predictions_flat[valid_mask]]
        true_labels = label_array[labels_flat[valid_mask]]

        metric = evaluate.load("seqeval")
        results = metric.compute(predictions=[true_predictions.tolist()], references=[true_labels.tolist()], zero_division=0)

        msg = f"precision:{results['overall_precision']}|recall:{results['overall_recall']}|f1:{results['overall_f1']}|accuracy:{results['overall_accuracy']}"
        ugly_log(log_file, msg)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    except Exception as e:
        ugly_log(log_file, f"Error in compute_metrics: {str(e)}")
        return {}
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" not in inputs or inputs["labels"] is None:
            ugly_log(self.args.log_file, "Labels are missing or None in compute_loss")
            if return_outputs:
                return torch.tensor(0.0), None
            return torch.tensor(0.0)
        # Check if reweighting is enabled and a meta optimizer is provided
        if getattr(self.args, 'reweight', False) and hasattr(self, 'meta_optimizer'):
            labels = inputs.pop("labels")  # Extract labels from inputs for custom handling
            inputs = {key: val.to(self.args.device) for key, val in inputs.items()}  # Ensure all tensors are on the correct device
            
            with higher.innerloop_ctx(model, self.meta_optimizer) as (fmodel, diffopt):
                # Forward pass on the pseudo model
                outputs = fmodel(**inputs, labels=labels)
                meta_train_loss = outputs.loss.mean()

                # Creating a differentiable copy of meta_train_loss
                eps = torch.zeros_like(meta_train_loss, requires_grad=True).to(self.args.device)
                modified_loss = meta_train_loss + eps
                diffopt.step(modified_loss)

                # Use actual data to compute meta-validation loss
                meta_outputs = fmodel(**inputs, labels=labels)
                meta_val_loss = meta_outputs.loss.mean()

                # Compute gradients of eps with respect to meta-validation loss
                eps_grad = torch.autograd.grad(meta_val_loss, eps, only_inputs=True)[0]

                # Reweight the losses based on computed gradients
                reweighted_loss = torch.sum(eps_grad * outputs.loss)

            # Proceed with the standard backward pass using reweighted loss
            if return_outputs:
                return reweighted_loss, outputs
            return reweighted_loss
        else:
            # Standard loss computation if no reweighting is enabled
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

def train_ner(args, train_dataset, dev_dataset, model, id2label, tokenizer):
    train_dataset = filter_none_labels(train_dataset)
    dev_dataset = filter_none_labels(dev_dataset)
    
    max_index = max(id2label.keys())
    label_array = np.array([id2label[i] for i in range(max_index + 1)])

    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics_with_id2label = partial(compute_metrics, label_array=label_array, log_file=args.log_file)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=3,  # Keep only the most recent best model according to the evaluation
        load_best_model_at_end=True,  # Load the best model at the end of training
        weight_decay=0.01,
        seed=args.seed,
        push_to_hub=True,
        hub_model_id=args.model_name_on_hub,
    )

    setattr(training_args, 'reweight', True)  # or False, depending on your requirement
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_with_id2label
    )
    trainer.train()
    return trainer.model


