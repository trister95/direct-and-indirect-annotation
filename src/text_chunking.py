import nltk
from nltk.tokenize import sent_tokenize
import sys
import argparse

def fix_short_sentences(text):
    sentences = sent_tokenize(text)
    adjusted_sentences = []
    i = 0

    while i < len(sentences):
        words = sentences[i].split()
        while len(words) < 10 and i + 1 < len(sentences):
            i += 1
            words += sentences[i].split()
        i += 1
        adjusted_sentences.append(' '.join(words))
    return adjusted_sentences

def split_list_equally(lst, n_parts):
    part_size = len(lst) // n_parts
    remainder = len(lst) % n_parts
    
    split_indices = [part_size] * n_parts
    for i in range(remainder):
        split_indices[i] += 1
    
    current_index = 0
    result = []
    for size in split_indices:
        result.append(lst[current_index:current_index + size])
        current_index += size
    return [" ".join(e) for e in result]

def fix_long_sentences(sentences):
    new_sentences = []
    for s in sentences:
        words = s.split()
        length = len(words)
        if length <= 100:
            new_sentences.append(s)
        if length > 100:
            no_of_splits = length // 100 + 1
            split_sentences = split_list_equally(words, no_of_splits)
            for spl_sent in split_sentences:
                new_sentences.append(spl_sent)
    return new_sentences

def main(input_file, output_file):
    nltk.download('punkt', quiet=True)

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except IOError:
        print(f"Error reading file: {input_file}")
        sys.exit(1)

    text = text.replace('\n', ' ')
    while "  " in text:
        text = text.replace("  ", " ")

    fixed_for_short_sentences = fix_short_sentences(text)
    fixed_for_long_sentences = fix_long_sentences(fixed_for_short_sentences)

    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for sentence in fixed_for_long_sentences:
                file.write(sentence + '\n')
    except IOError:
        print(f"Error writing to file: {output_file}")
        sys.exit(1)

    print(f"Processing complete. Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text to fix short and long sentences.")
    parser.add_argument('input_file', type=str, help='The input text file to process.')
    parser.add_argument('output_file', type=str, help='The output file to save processed text.')

    args = parser.parse_args()

    main(args.input_file, args.output_file)
