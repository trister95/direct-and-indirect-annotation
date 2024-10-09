from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils import ner_collate_fn

from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    # Convert list to tensor before passing to default_collate
    batch = [{k: torch.tensor(v) if isinstance(v, list) else v for k, v in b.items()} for b in batch]
    return default_collate(batch)

def ner_predict(args, model, dataset):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=ner_collate_fn) #or: collate

    pred_logits = []
    max_len = 0  # Initialize max_len

    # First loop to determine the global maximum length
    for batch in data_loader:
        with torch.no_grad():
            inputs = {'input_ids': batch['input_ids'].to(model.device),
                      'attention_mask': batch['attention_mask'].to(model.device)}
            outputs = model(**inputs)
            logits = outputs.logits
            current_max = max((inputs['attention_mask'] == 1).sum(dim=1)).item()
            if current_max > max_len:
                max_len = current_max

    # Second loop to process and pad data
    for batch in tqdm(data_loader, desc='Evaluating on pool data'):
        with torch.no_grad():
            inputs = {'input_ids': batch['input_ids'].to(model.device),
                      'attention_mask': batch['attention_mask'].to(model.device)}
            outputs = model(**inputs)
            logits = outputs.logits

            for i in range(logits.size(0)):
                active_logits = logits[i][inputs['attention_mask'][i] == 1]
                # Pad to global max_len
                padded_logits = torch.nn.functional.pad(active_logits, (0, 0, 0, max_len - active_logits.size(0)))
                pred_logits.append(padded_logits)

    # Convert list of tensors to a tensor
    pred_logits = torch.stack(pred_logits)
    return pred_logits

def compute_uncertainty(logits, reduction='mean'):
    probs = torch.softmax(torch.tensor(logits), dim=-1)
    confidence = torch.max(probs, dim=-1)[0]
    uncertainty = 1 - confidence
    if reduction == 'mean':
        return uncertainty.mean(dim=1)
    elif reduction == 'sum':
        return uncertainty.sum(dim=1)
    elif reduction == 'min':
        return uncertainty.max(dim=1)
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")


def get_bert_embeddings(args, features, model: nn.Module, normalize: bool=True):
    """
    embed features with bert encoder.
    """
    model.eval()
    embeddings = []
    if isinstance(model, AutoModelForTokenClassification):
        dataloader = DataLoader(features, args.test_batch_size, shuffle=False, collate_fn=ner_collate_fn)
    else:
        raise ValueError('Unknown model.')
    for batch in tqdm(dataloader, desc='Computing bert embeddings'):
        inputs = {'input_ids': batch['input_ids'].to(args.device),
                  'attention_mask': batch['attention_mask'].to(args.device)}    
        with torch.no_grad():
            outputs = model.bert(**inputs)[0]   # [batch_size, seq_length, n_dim]
            attention_mask = inputs['attention_mask']
            outputs[attention_mask == 0] = 0
            # mean pooling over sequence outputs
            outputs = outputs.sum(dim=1) / attention_mask.sum(dim=-1).unsqueeze(-1)
            if normalize:
                outputs = F.normalize(outputs, p=2, dim=-1)    
            embeddings.append(outputs)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings