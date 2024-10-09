import os
import json
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from .ner_reader import ner_reader

class Processor:
    def __init__(self, dataset: str, tokenizer: AutoTokenizer, cache_name: str = 'cache'):
        self.tokenizer = tokenizer
        self.dataset_name = dataset
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(os.path.dirname(dir_path))
        dir_path = os.path.join(dir_path, f'data/{dataset}')
        
        meta_path = os.path.join(dir_path, 'meta.json')
        self.tag2id = json.load(open(meta_path, 'r'))['tag2id']
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        if cache_name is None:
            self.cache_file = None
            self.cache_name = None
            self.use_cache = False

        else:
            self.cache_file = os.path.join(dir_path, f'{cache_name}.json')  
            self.cache_name = cache_name
            self.use_cache = True
            
            if not os.path.exists(self.cache_file):
                cache = dict()
                json.dump(cache, open(self.cache_file, 'w'))
                
        print(f'Loading dataset {dataset} ...')
        
        data = ner_reader(tokenizer, self.dataset_name, self.cache_name,
                          self.use_cache)
              
        self.tokenized_datasets = DatasetDict({
            split: Dataset.from_dict(data[split])
            for split in data
            })  
        print('Finish loading dataset.')

    def update_cache(self, records: dict):
        cache = json.load(open(self.cache_file, 'r', encoding='utf-8'))
        cache.update(records)
        json.dump(cache, open(self.cache_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    
    def reload(self):
        print('Reloading dataset...')
        data = ner_reader(self.tokenizer, self.dataset_name, self.cache_name, self.use_cache) #changed this from features to tokenized_datasets
        self.tokenized_datasets = DatasetDict({
          split: Dataset.from_dict(data[split])
          for split in data})  
        print('Finish reloading dataset.')

    def get_id2tag(self):
        return self.id2tag
    
    def get_tag2id(self):
        return self.tag2id
    
    def get_features(self, split: str):
        if split in self.tokenized_datasets:
            #if split == 'train':
            return self.tokenized_datasets[split]
        else:
            raise ValueError(f"Split '{split}' not found in dataset")

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    
    # Define the dataset name
    dataset_name = 'en_conll03'
    
    # Assuming the directory structure and meta.json location as per your setup
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(os.path.dirname(dir_path))
    dataset_dir = os.path.join(dir_path, f'data/{dataset_name}')

    # Load the tag2id dictionary from meta.json
    meta_path = os.path.join(dataset_dir, 'meta.json')
    with open(meta_path, 'r') as file:
        tag2id = json.load(file)['tag2id']

    # Now pass tag2id to the ner_reader
    dataset = ner_reader(tokenizer, dataset_name)
    print(dataset['train'][0])
