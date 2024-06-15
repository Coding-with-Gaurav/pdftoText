import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from PyPDF2 import PdfReader
from collections import Counter, defaultdict
import re
from itertools import combinations

nltk.download('punkt')
nltk.download('stopwords')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_by_page = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            text_by_page.append((page_number, text))
    return text_by_page

def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text saved to file: {file_path}")

def tokenize_text(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if re.sub(r'[^\w\s]', '', token)]
    return cleaned_tokens

def remove_stopwords_and_punctuation(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
    return filtered_tokens

def apply_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def save_ner_to_file(entities, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entity, label in entities:
            f.write(f"{entity} ({label})\n")
    print(f"NER results saved to file: {file_path}")

def count_word_frequency(tokens):
    frequency = Counter(tokens)
    return frequency

def save_frequency_to_file(frequency, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for word, count in frequency.items():
            f.write(f"{word}: {count}\n")
    print(f"Word frequency saved to file: {file_path}")

def compare_files(file1_path, file2_path, output_file_path):
    with open(file1_path, encoding='utf-8') as file1, open(file2_path, encoding='utf-8') as file2:
        tokens1 = file1.read().split()
        tokens2 = file2.read().split()
        
        frequency1 = count_word_frequency(tokens1)
        frequency2 = count_word_frequency(tokens2)
        
        common_words = set(frequency1.keys()) & set(frequency2.keys())
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for word in common_words:
                f.write(f"{word}: file1({frequency1[word]}), file2({frequency2[word]})\n")
        print(f"Common word frequencies saved to file: {output_file_path}")

def process_pdf(pdf_path, base_output_path):
    text_file_path = os.path.join(base_output_path, f"{os.path.basename(pdf_path).replace('.pdf', '')}_extracted_text.txt")
    tokenized_file_path = os.path.join(base_output_path, f"{os.path.basename(pdf_path).replace('.pdf', '')}_tokenized_text.txt")
    ner_file_path = os.path.join(base_output_path, f"{os.path.basename(pdf_path).replace('.pdf', '')}_ner_results.txt")
    frequency_file_path = os.path.join(base_output_path, f"{os.path.basename(pdf_path).replace('.pdf', '')}_word_frequency.txt")
    page_info_path = os.path.join(base_output_path, f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_info.txt")

    text_by_page = extract_text_from_pdf(pdf_path)
    all_text = " ".join([text for page, text in text_by_page])
    save_text_to_file(all_text, text_file_path)

    tokens_by_page = [(page, tokenize_text(text)) for page, text in text_by_page]
    all_tokens = [token for page, tokens in tokens_by_page for token in tokens]
    save_text_to_file(' '.join(all_tokens), tokenized_file_path)

    filtered_tokens_by_page = [(page, remove_stopwords_and_punctuation(tokens)) for page, tokens in tokens_by_page]
    all_filtered_tokens = [token for page, tokens in filtered_tokens_by_page for token in tokens]

    ner_entities = apply_ner(' '.join(all_filtered_tokens))
    save_ner_to_file(ner_entities, ner_file_path)

    frequency = count_word_frequency(all_filtered_tokens)
    save_frequency_to_file(frequency, frequency_file_path)

    word_page_mapping = defaultdict(list)
    for page, tokens in filtered_tokens_by_page:
        for token in tokens:
            word_page_mapping[token].append(page)
    
    with open(page_info_path, 'w', encoding='utf-8') as f:
        for word, pages in word_page_mapping.items():
            f.write(f"{word}: {pages}\n")

    return tokenized_file_path, page_info_path

def compare_files_with_page_info(file1_path, file2_path, page_info1_path, page_info2_path, output_file_path):
    with open(file1_path, encoding='utf-8') as file1, open(file2_path, encoding='utf-8') as file2:
        tokens1 = file1.read().split()
        tokens2 = file2.read().split()

        frequency1 = count_word_frequency(tokens1)
        frequency2 = count_word_frequency(tokens2)

        common_words = set(frequency1.keys()) & set(frequency2.keys())

        with open(page_info1_path, encoding='utf-8') as pi1, open(page_info2_path, encoding='utf-8') as pi2:
            page_info1 = {line.split(': ')[0]: eval(line.split(': ')[1]) for line in pi1.readlines()}
            page_info2 = {line.split(': ')[0]: eval(line.split(': ')[1]) for line in pi2.readlines()}

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for word in common_words:
                f.write(f"{word}: file1({frequency1[word]}), file2({frequency2[word]}), pages_file1({page_info1.get(word, [])}), pages_file2({page_info2.get(word, [])})\n")
        print(f"Common word frequencies and page numbers saved to file: {output_file_path}")

def main(pdf_paths, base_output_path):
    processed_files = []
    page_info_files = []
    for pdf_path in pdf_paths:
        tokenized_file, page_info_file = process_pdf(pdf_path, base_output_path)
        processed_files.append(tokenized_file)
        page_info_files.append(page_info_file)
    
    for (file1, page_info1), (file2, page_info2) in combinations(zip(processed_files, page_info_files), 2):
        output_file_path = os.path.join(base_output_path, f"comparison_{os.path.basename(file1)}_{os.path.basename(file2)}.txt")
        compare_files_with_page_info(file1, file2, page_info1, page_info2, output_file_path)

pdf_paths = ['what_is_spirituality.pdf', 'Health_Tips_From_The_Vedas.pdf']
base_output_path = './output'
main(pdf_paths, base_output_path)
