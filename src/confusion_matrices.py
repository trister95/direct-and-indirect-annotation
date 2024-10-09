import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from itertools import product


class NERPredictor:
    """
    A custom NERPredictor is needed because the huggingface pipeline abstraction did cut off parts of the labels. 
    This can probably be prevented by not using "-" in the labels.
    """
    def __init__(self, model_name):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        filtered_tokens = [token for token in tokens if token not in self.tokenizer.all_special_tokens]
        filtered_predictions = [pred for token, pred in zip(tokens, predictions) if token not in self.tokenizer.all_special_tokens]
        return [self._get_most_common(labels) for _, labels in self._aggregate_tokens(filtered_tokens, filtered_predictions)]

    def _aggregate_tokens(self, tokens, predictions):
        word_predictions = defaultdict(list)
        current_word = ""
        for token, pred in zip(tokens, predictions):
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    yield current_word, word_predictions[current_word]
                    word_predictions[current_word] = []
                current_word = token
            word_predictions[current_word].append(self.id2label[pred.item()])
        if current_word:
            yield current_word, word_predictions[current_word]

    def _get_most_common(self, labels):
        return max(set(labels), key=labels.count)

def load_jsonl(file_path, key='tags'):
    results = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if key in data:
                    results.append(data[key])
                else:
                    print(f"Key '{key}' not found in line {i+1}: {line.strip()}")
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError in line {i+1}: {e.msg} (line: {line.strip()})")
            except Exception as e:
                print(f"Error processing line {i+1}: {e} (line: {line.strip()})")
    return results

def flatten_predictions(predictions):
    return [item for sublist in predictions for item in sublist]

def create_confusion_matrix(true_labels, pred_labels):
    def process_label(x):
        if x is None or x == "None":
            return 'No label'
        return 'No label' if x == 'O' else x.replace('-', ' ')

    true_labels = [process_label(label) for label in true_labels]
    pred_labels = [process_label(label) for label in pred_labels]
    
    all_labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    
    # Avoid division by zero with a small epsilon
    epsilon = 1e-8
    sum_with_epsilon = cm.sum(axis=1)[:, np.newaxis] + epsilon
    
    cm_normalized = cm.astype('float') / sum_with_epsilon
    
    return cm, cm_normalized, all_labels


def plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, output_folder, use_absolute_numbers=False, highlight_labels=None):
    plt.figure(figsize=(12, 10))  # Kept original figure size
    
    # Custom annotation function to replace zeros with empty string
    def fmt(x):
        if x == 0:
            return '-'
        if use_absolute_numbers:
            return f'{int(x)}'
        else:
            return f'{x:.2f}'
    
    # Create the heatmap
    if use_absolute_numbers:
        cm_int = np.round(cm).astype(int)
        annot = np.vectorize(fmt)(cm_int)
        ax = sns.heatmap(cm_normalized, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels,
                         annot_kws={'size': 16})
    else:
        annot = np.vectorize(fmt)(cm_normalized)
        ax = sns.heatmap(cm_normalized, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels,
                         annot_kws={'size': 16})
    
    # Adjust colorbar to include '1' at the top
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    colorbar.ax.tick_params(labelsize=10)  # Kept increased colorbar tick label size
    
    # Highlight specific cells if requested
    if highlight_labels:
        print(f"Highlighting label intersections: {highlight_labels}")  # Debug print
        for true_label, pred_label in highlight_labels:
            if true_label in labels and pred_label in labels:
                true_idx = labels.index(true_label)
                pred_idx = labels.index(pred_label)
                rect = plt.Rectangle((pred_idx, true_idx), 1, 1, fill=False, edgecolor='red', lw=3)
                ax.add_patch(rect)
            else:
                print(f"Warning: Label pair ({true_label}, {pred_label}) not found in the matrix")
    else:
        print("No cells to highlight")  # Debug print
    
    #plt.title(f'Confusion Matrix: {method1} vs {method2}')  # Kept original title font size
    plt.xlabel(f'Predicted ({method2})', fontsize=14)  # Kept increased x-label font size
    plt.ylabel(f'True ({method1})', fontsize=14)  # Kept increased y-label font size
    plt.xticks(fontsize=12, rotation=45, ha='right')  # Kept increased x-tick label font size and rotation
    plt.yticks(fontsize=12)  # Kept increased y-tick label font size
    plt.tight_layout()
    output_file = os.path.join(output_folder, f'confusion_matrix_{method1}_vs_{method2}.png')
    plt.savefig(output_file, dpi=600)  # Kept increased DPI for better resolution
    plt.close()

def main(args):
    D_hum_predictions = load_jsonl(args.d_human_annotations, 'tags')
    T_hum_predictions = load_jsonl(args.t_human_annotations, 'tags')
    texts = load_jsonl(args.d_human_annotations, 'text')
    D_predictor = NERPredictor(args.d_model)
    T_predictor = NERPredictor(args.t_model)
    D_lllmaaa_predictions = [D_predictor.predict(text) for text in texts]
    T_lllmaaa_predictions = [T_predictor.predict(text) for text in texts]
    data = {
        "human1": D_hum_predictions,
        "human2": T_hum_predictions,
        "direct_zeroshot": load_jsonl(args.direct_zeroshot_file, 'tags'),
        "direct_fewshot1": load_jsonl(args.direct_demo1_file, 'tags'),
        "direct_fewshot2": load_jsonl(args.direct_demo2_file, 'tags'),
        "indirect1": D_lllmaaa_predictions,
        "indirect2": T_lllmaaa_predictions,
    }
    os.makedirs(args.output_folder, exist_ok=True)
    for method1, method2 in product(data.keys(), repeat=2):
        if method1 != method2:
            print(f"Generating confusion matrix for {method1} vs {method2}")
            flat_preds1 = flatten_predictions(data[method1])
            flat_preds2 = flatten_predictions(data[method2])
            try:
                cm, cm_normalized, labels = create_confusion_matrix(flat_preds1, flat_preds2)
                # Determine how to plot based on output_type
                if args.output_type == 'both':
                    plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, args.output_folder, use_absolute_numbers=True, highlight_labels=args.highlight_labels)
                    plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, args.output_folder, use_absolute_numbers=False, highlight_labels=args.highlight_labels)
                elif args.output_type == 'absolute':
                    plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, args.output_folder, use_absolute_numbers=True, highlight_labels=args.highlight_labels)
                elif args.output_type == 'relative':
                    plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, args.output_folder, use_absolute_numbers=False, highlight_labels=args.highlight_labels)
            except Exception as e:
                print(f"Error creating confusion matrix: {e}")
    print(f"Confusion matrices have been saved to '{args.output_folder}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Confusion Matrix Generator")
    parser.add_argument("--d_human_annotations", required=True, help="Path to D human annotations file")
    parser.add_argument("--t_human_annotations", required=True, help="Path to T human annotations file")
    parser.add_argument("--d_model", default="ArjanvD95/by_the_horns_D42G", help="Path to D model")
    parser.add_argument("--t_model", default="ArjanvD95/by_the_horns_T42G", help="Path to T model")
    parser.add_argument("--direct_zeroshot_file", default = "data/by_the_horns_D/predictions_no_demo/pred1.jsonl", help="Path to file containing direct_demo1 predictions")
    parser.add_argument("--direct_demo1_file", required=True, help="Path to file containing direct_demo1 predictions")
    parser.add_argument("--direct_demo2_file", required=True, help="Path to file containing direct_demo2 predictions")
    parser.add_argument("--output_folder", default="confusion_matrices", help="Path to output folder for confusion matrices")
    parser.add_argument("--output_type", choices=['relative', 'absolute', 'both'], default='both', help="Type of confusion matrix to output")
    parser.add_argument("--highlight_labels", nargs=2, action='append', help="Pair of labels to highlight. Use multiple --highlight_labels arguments for multiple intersections.")
    args = parser.parse_args()
    
    print(f"Label intersections to highlight: {args.highlight_labels}")  # Debug print
    
    main(args)