import sys
from lxml import etree


def extract_text(tree):
    texts = []
    for div in tree.xpath('//body//div[@type="chapter"]'):
        chapter_text = ''.join(div.itertext())
        texts.append(chapter_text.strip())
    return "\n\n".join(texts)

def main(xml_file_path):
    try:
        # Load the XML file
        with open(xml_file_path, 'r', encoding='utf-8') as file:
            xml_content = file.read()

        # Parse the XML file using lxml
        tree = etree.fromstring(xml_content.encode('utf-8'))

        # Extract text from the XML
        book_text = extract_text(tree)

        # Define the output file path
        output_file_path = xml_file_path.replace('.xml', '_extracted.txt')

        # Save the extracted text to a file
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(book_text)

        print(f"Text extracted and saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_book_text.py <path_to_xml_file>")
    else:
        xml_file_path = sys.argv[1]
        main(xml_file_path)