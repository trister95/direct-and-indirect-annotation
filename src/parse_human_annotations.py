import argparse
import cassis
import json
import os
import re
import random
from typing import List, Dict

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

def process_annotations(base_folder: str, output_file: str):
    all_documents_sentences = []

    for document_name in os.listdir(base_folder):
        document_folder = os.path.join(base_folder, document_name)
        xml_file = next(os.path.join(document_folder, f) for f in os.listdir(document_folder) if f.endswith('.xml'))
        xmi_file = next(os.path.join(document_folder, f) for f in os.listdir(document_folder) if f.endswith('.xmi'))

        with open(xml_file, 'rb') as f:
            typesystem = cassis.load_typesystem(f)

        with open(xmi_file, 'rb') as f:
            cas = cassis.load_cas_from_xmi(f, typesystem=typesystem)

        SentenceType = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence')
        NamedEntityType = typesystem.get_type('custom.Span')

        sentences_list = []

        for sentence in cas.select(SentenceType):
            sentence_text = cas.sofa_string[sentence.begin:sentence.end]
            labels = []
            
            for named_entity in cas.select_covered(NamedEntityType, sentence):
                label_text = cas.sofa_string[named_entity.begin:named_entity.end]
                labels.append({
                    "text": label_text,
                    "start": named_entity.begin - sentence.begin, 
                    "end": named_entity.end - sentence.begin,
                    "label": getattr(named_entity, 'label', 'Unknown')
                })
            
            sentences_list.append({
                "text": sentence_text,
                "labels": labels
            })

        all_documents_sentences.extend(sentences_list)

    with open(output_file, 'w') as f:
        json.dump(all_documents_sentences, f, indent=4)

    print(f"All annotations from documents have been saved to {output_file}")

def tokenize(text: str) -> List[str]:
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

def create_tags(tokens: List[str], span_label: List[Dict]) -> List[str]:
    if span_label:
        for e in span_label:
            e["span"] = e["text"]
            e["type"] = e["label"]
    span_label = sorted(span_label, key=lambda x: len(x['span']), reverse=True)
    span_to_type = {entity['span']: entity['type'] for entity in span_label}

    dictionary = {token: f'[{i}]' for i, token in enumerate(set(tokens))}
    id_string = ' '.join(dictionary[token] for token in tokens)

    for entity in span_label:
        span_tokens = entity['span'].strip().split()
        if not all(token in dictionary for token in span_tokens):
            continue
        id_substring = ' '.join(dictionary[token] for token in span_tokens)
        id_string = f'[sep]{id_substring}[sep]'.join(id_string.split(id_substring))

    sent = id_string
    for token, id_token in dictionary.items():
        sent = sent.replace(id_token, token)
    words = sent.split('[sep]')

    seq_label = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        entity_flag = word in span_to_type
        word_length = len(word.split())
        if entity_flag:
            label = [span_to_type[word]] * word_length
        else:
            label = ['O'] * word_length
        seq_label.extend(label)

    assert len(seq_label) == len(tokens)
    return seq_label 

def transform_annotations(input_annotations: List[Dict]) -> List[Dict]:
    output_data = []
    for idx, annotation in enumerate(input_annotations):
        text = annotation['text']
        labels = annotation['labels']
        
        updated_labels = [
            {'text': label['text'], 'label': LABEL_MAPPING.get(label['label'], label['label'])}
            for label in labels
        ]

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

def process_file(input_f: str, output_f: str):
    with open(input_f, 'r', encoding='utf-8') as infile:
        input_data = json.load(infile)

    transformed_data = transform_annotations(input_data)

    with open(output_f, 'w', encoding='utf-8') as outfile:
        for item in transformed_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data: List[Dict], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def split_data(input_f: str, demo_f: str, test_f: str, holdout_f: str):
    data = load_jsonl(input_f)

    random.seed(42)
    random.shuffle(data)

    demo_data = data[0:100]
    test_data = data[100:150]
    holdout_data = data[150:]

    save_jsonl(demo_data, demo_f)
    save_jsonl(test_data, test_f)
    save_jsonl(holdout_data, holdout_f)

    print(f'Demo data saved to: {demo_f}')
    print(f'Test data saved to: {test_f}')
    print(f'Holdout data saved to: {holdout_f}')

def main():
    parser = argparse.ArgumentParser(description="Process and split annotation data")
    parser.add_argument("--base_folder", required=True, help="Base folder containing XML and XMI files")
    parser.add_argument("--output_file", required=True, help="Output file for processed annotations")
    parser.add_argument("--demo_file", required=True, help="Output file for demo data")
    parser.add_argument("--test_file", required=True, help="Output file for test data")
    parser.add_argument("--holdout_file", required=True, help="Output file for holdout data")

    args = parser.parse_args()

    # Process annotations
    process_annotations(args.base_folder, args.output_file)

    # Transform annotations
    intermediate_file = 'intermediate.jsonl'
    process_file(args.output_file, intermediate_file)

    # Split data
    split_data(intermediate_file, args.demo_file, args.test_file, args.holdout_file)

    # Clean up intermediate file
    os.remove(intermediate_file)

if __name__ == "__main__":
    main()