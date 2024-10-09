import ujson as json
import os
import re
import argparse
from func_timeout import func_set_timeout
from openai import RateLimitError
from langchain_core import prompts, output_parsers
from langchain_openai import ChatOpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

LABEL_MAPPING = {
    "An-Org-Lit": "Animals-Organisms-Literal",
    "An-Org-Sym": "Animals-Organisms-Symbolical",
    "An-Org-Petrified": "Animals-Organisms-Petrified",
    "An-Part-Lit": "Animals-Parts-Literal",
    "An-Part-Sym": "Animals-Parts-Symbolical",
    "An-Part-Petrified": "Animals-Parts-Petrified",
    "An-Prod-Lit": "Animals-Products-Literal",
    "An-Prod-Sym": "Animals-Products-Symbolical",
    "An-Prod-Petrified": "Animals-Products-Petrified",
    "An-Coll-Lit": "Animals-Collective-Literal",
    "An-Coll-Sym": "Animals-Collective-Symbolical",
    "An-Coll-Petrified": "Animals-Collective-Petrified",
    "Plant-Org-Lit": "Plants-Organisms-Literal",
    "Plant-Org-Sym": "Plants-Organisms-Symbolical",
    "Plant-Org-Petrified": "Plants-Organisms-Petrified",
    "Plant-Part-Lit": "Plants-Parts-Literal",
    "Plant-Part-Sym": "Plants-Parts-Symbolical",
    "Plant-Part-Petrified": "Plants-Parts-Petrified",
    "Plant-Prod-Lit": "Plants-Products-Literal",
    "Plant-Prod-Sym": "Plants-Products-Symbolical",
    "Plant-Prod-Petrified": "Plants-Products-Petrified",
    "Plant-Coll-Literal": "Plants-Collective-Literal",
    "Plant-Coll-Sym": "Plants-Collective-Symbolical",
    "Plant-Coll-Petrified": "Plants-Collective-Petrified"
}

def tokenize(text: str) -> List[str]:
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

def create_tags(tokens, span_label):
    """
    Covert span labels to sequence labels.
    """
    if span_label != []:
        for e in span_label:
            e["span"] = e["text"]
            e["type"] = e["label"]
    span_label = sorted(span_label, key=lambda x: len(x['span']), reverse=True)
    span_to_type = {entity['span']: entity['type'] for entity in span_label}
    # get words list

    # build a tokenizer first
    dictionary = dict()
    for token in tokens:
        if token not in dictionary:
            dictionary[token] = f'[{len(dictionary)}]'
    id_string = ' '.join([dictionary[token] for token in tokens])
    for entity in span_label:
        span_tokens = entity['span'].strip().split(' ')
        # validate span token
        valid_flag = True
        for token in span_tokens:
            if token not in dictionary:
                valid_flag = False
                break
        if not valid_flag:
            continue
        # translate span token into ids
        id_substring = ' '.join([dictionary[token] for token in span_tokens])
        id_string = ('[sep]' + id_substring + '[sep]').join(id_string.split(id_substring))
        # print(id_string)
    # convert back to nl
    sent = id_string
    for token in dictionary:
        sent = sent.replace(dictionary[token], token)
    words = sent.split('[sep]')

    seq_label = []
    for word in words:
        word = word.strip()
        if len(word) == 0:
            continue
        entity_flag = (word in span_to_type)
        word_length = len(word.split(' '))
        if entity_flag:
            if word_length == 1:
                label = [f'{span_to_type[word]}']
            else:
                label = ([f'{span_to_type[word]}'] * (word_length))
        else:
            label = ['O' for _ in range(word_length)]
        seq_label.extend(label)

    assert len(seq_label) == len(tokens)
    return seq_label 

def transform_annotations(input_annotations: List[Dict]) -> List[Dict]:
    output_data = []
    for idx, annotation in enumerate(input_annotations):
        text = annotation['text']
        labels = annotation['labels']
        
        updated_labels = [{'text': label['span'], 'label': LABEL_MAPPING.get(label['type'], label['type'])} 
                          for label in labels]

        tokens = tokenize(text)
        tags = create_tags(tokens, updated_labels)

        labels_list = [{'span': label['text'], 'type': label['label']} for label in updated_labels]

        transformed_annotation = {
            'tokens': tokens,
            'tags': tags,
            'text': text,
            'labels': labels_list,
            'id': str(idx)
        }
        output_data.append(transformed_annotation)
    return output_data

class Annotator:
    def __init__(self, engine: str = 'gpt-4o', config_path: str = 'default', dataset: str = None, use_demos: bool = True):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)

        self.dataset = dataset or config['dataset']
        self.task = config['task']
        self.description = config['description']
        self.guidance = config['guidance']
        self.input_format = config['input_format']
        self.output_format = config['output_format']
        self.struct_format = config['struct_format']

        self.use_demos = use_demos
        demo_file_path = os.path.join(f'data/{self.dataset}/demo.jsonl')
        self.demo_file = {}
        with open(demo_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.demo_file[sample['id']] = sample
        demo_index_path = os.path.join(f'data/{self.dataset}/train-knn-demo.json')
        with open(demo_index_path, 'r', encoding='utf-8') as f:
            self.demo_index = json.load(f)

        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=engine)

        self.prompt_template = prompts.ChatPromptTemplate.from_messages([
            ("system", self.description.replace("{", "{{").replace("}", "}}")),
            ("system", self.guidance),
            ("user", "{input}")
        ])

        self.output_parser = output_parsers.StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser

    def prepare_demo(self, sample_id: str) -> List[Dict]:
        return [self.demo_file[pointer['id']] for pointer in reversed(self.demo_index.get(sample_id, []))]

    def generate_prompt(self, sample: Dict, demo: List[Dict] = None) -> str:
        to_annotate = self.input_format.format(json.dumps(sample['text']))
        if self.use_demos and demo:
            demo_annotations = "\n".join(
                f"{self.input_format.format(json.dumps(d['text']))}\n{self.output_format.format(json.dumps(d['labels']))}" for d in demo
            )
            return f"Here are some examples:\n{demo_annotations}\n\nPlease now annotate the following input:\n{to_annotate}"
        return f"Please annotate the following input:\n{to_annotate}" 
    
    @func_set_timeout(60)
    def online_annotate(self, sample: Dict) -> List[Dict]:
        demo = self.prepare_demo(sample['id']) if self.use_demos else None
        annotation_prompt = self.generate_prompt(sample, demo)
                
        for _ in range(3):  # Allow up to 3 attempts
            try:
                response = self.chain.invoke({"input": annotation_prompt})
                return self.postprocess(response)
            except RateLimitError:
                print("Rate limit exceeded. Please wait and try again.")
                print(f"Problem was with: {annotation_prompt}")
                return None
            except Exception as e:
                print(f"Error during annotation: {e}")
                print(f"Problem was with: {annotation_prompt}")
        
        print("Max retries reached. Aborting operation.")
        return None
    
    def postprocess(self, result: str) -> List[Dict]:
        meta_path = f"data/{self.dataset}/meta.json"
        with open(meta_path, 'r') as file:
            meta = json.load(file)
        tagset = meta['tagset']
        
        match = re.search(r"\[([^\]]*)\]", result)
        extracted_result = eval(f"[{match.group(1)}]") if match and match.group(1) else []

        return [entity for entity in extracted_result 
                if isinstance(entity, dict) and 'type' in entity and 'span' in entity and entity['type'] in tagset]

    def online_annotate_and_transform(self, sample: Dict) -> Dict:
        annotation = self.online_annotate(sample)
        if annotation is None:
            return None
      
        transformed_annotation = transform_annotations([{
            'text': sample['text'],
            'labels': annotation
        }])
        return transformed_annotation[0]

def process_annotations_file(to_annotate_path: str, annotator: Annotator, output_folder: str, n: int):
    for i in range(1, n + 1):
        output_file_path = os.path.join(output_folder, f'pred{i}.jsonl')
        with open(to_annotate_path, 'r', encoding='utf-8') as file, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line in file:
                sample = json.loads(line.strip())
                transformed_annotation = annotator.online_annotate_and_transform(sample)
                if transformed_annotation:
                    json.dump(transformed_annotation, outfile, ensure_ascii=False)
                    outfile.write('\n')
        print(f"Annotation {i} complete. Output saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Annotation Script")
    parser.add_argument("--to_annotate_path", type=str, required=True, help="Path to the file to annotate")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the output files")
    parser.add_argument("--engine", type=str, default="gpt-4o", help="GPT engine to use (default: gpt-4o)")
    parser.add_argument("--n", type=int, required=True, help="Number of times to run the annotation")
    parser.add_argument("--use_demos", action="store_true", help="Use demonstrations for annotation")


    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    annotator = Annotator(engine=args.engine, config_path=args.config_path, dataset=args.dataset, use_demos=args.use_demos)
    process_annotations_file(args.to_annotate_path, annotator, args.output_folder, args.n)

    print(f"Annotation complete. Outputs saved to {args.output_folder}")
