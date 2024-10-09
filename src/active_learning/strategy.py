import os
import time
import ujson as json
from func_timeout.exceptions import FunctionTimedOut
from openai import RateLimitError
from abc import ABC, abstractmethod
import numpy as np
from ..llm_annotator import Annotator

RETRY = 3

class Strategy(ABC):
    def __init__(self, annotator_config_name, pool_size, setting: str='knn', engine: str='gpt-3.5'):
        self.lab_data_mask = np.zeros(pool_size, dtype=bool)
        self.annotator = Annotator(engine, annotator_config_name)
        self.dataset = self.annotator.dataset
        if self.dataset in ['en_conll03', 'planimals', 'animals_or_not', 'by_the_horns_D', 'by_the_horns_T']:
            self.task_type = 'ner'
        else:
            raise ValueError('Unknown dataset.')
        self.setting = setting
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(os.path.dirname(dir_path))
        demo_file_path = os.path.join(dir_path, f'data/{self.dataset}/demo.jsonl')
        self.demo_file = dict()
        with open(demo_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.demo_file[sample['id']] = sample
        if setting == 'random' or setting == 'knn':
            demo_index_path = os.path.join(dir_path, f'data/{self.dataset}/train-{setting}-demo.json')
            self.demo_index = json.load(open(demo_index_path, 'r', encoding='utf-8'))
        elif setting == 'zero':
            pass
        else:
            raise ValueError(f'Unknown setting {setting}.')

    def __len__(self):
        return len(self.lab_data_mask)

    def _get_labeled_indices(self):
        return np.where(self.lab_data_mask)[0]
    
    def _get_pool_indices(self):
        return np.where(~self.lab_data_mask)[0]
    
    def get_labeled_data(self, features):
        labeled_indices = self._get_labeled_indices()
        labeled_data = features.select(labeled_indices)
        return labeled_data
    
    @abstractmethod
    def query(self, args, k, model, features, samples_likely_to_have_entity, k_enriched):
        return
    
    def init_labeled_data(self, n_sample: int = None, samples_likely_to_have_entity = [], n_enriched_indices = 0):
        if n_sample is None:
            raise ValueError('Please specify initial sample ratio/size.')
        if n_sample > len(self):
            raise ValueError('Initial sample size cannot be greater than the length of the data.')

        indices = np.arange(len(self))
        np.random.shuffle(indices)

        a = np.array(samples_likely_to_have_entity)[:n_enriched_indices]
        b = indices[:(n_sample - n_enriched_indices)]

        # Ensure both a and b are 1-dimensional arrays
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("Mismatch in array dimensions or data types")

        # Concatenate the arrays correctly
        combined_indices = np.concatenate((a, b))

        # Update the labeled data mask
        self.lab_data_mask[combined_indices] = True

        return combined_indices
    
    async def get_enriched_data(self, features, n_needed: int):
        if n_needed is None:
            raise ValueError('Please specify initial sample ratio/size.')
        if n_needed > len(self):
            raise ValueError('Sample size cannot be greater than the data size.')

        indices = np.arange(len(self))
        np.random.shuffle(indices)

        batch_size = 20  # Fixed batch size
        wanted_indices = []

        # Process data in batches of 20
        for i in range(0, len(indices), batch_size):
            if len(wanted_indices) >= n_needed:
                break  # Stop if we already have the required number of samples

            batch_features = [features[int(idx)] for idx in indices[i: i + batch_size]]
            batch_results = await self.annotator.process_batch_llm_async(batch_features)  # Adjust as needed

            # Check results and collect indices of true judgements
            for j, judgement in enumerate(batch_results):
                if judgement:
                    wanted_indices.append(indices[i + j])
                    if len(wanted_indices) >= n_needed:
                        break  # Stop collecting if we reach the required number

        # Trim the wanted_indices if they exceed n_sample
        if len(wanted_indices) > n_needed:
            wanted_indices = wanted_indices[:n_needed]
        return wanted_indices
   

    def update(self, indices, features):
        self.lab_data_mask[indices] = True
        records = self.annotate(features)
        return records
    
    def annotate(self, features):
        results = {}
        labeled_indices = self._get_labeled_indices()
        for i in labeled_indices:
            feature = features[int(i)]
            label_key = 'labels' if self.task_type == 'ner' else 'label_id'

            if feature[label_key] is None:
                if self.setting == 'random' or self.setting == 'knn':
                    demo = [self.demo_file[pointer['id']] for pointer in reversed(self.demo_index[feature['id']])]
                else:
                    demo = None

                result = None
                for j in range(RETRY):
                    try:
                        result = self.annotator.online_annotate(feature, demo)
                        break
                    except FunctionTimedOut:
                        print('Timeout. Retrying...')
                    except RateLimitError:
                        print('Rate limit. Sleep for 60 seconds...')
                        time.sleep(60)

                if result is None:
                    print(f"Error: No annotation result for index {i} (feature id {feature['id']}).")
                else:
                    results[feature['id']] = result
        return results
    
