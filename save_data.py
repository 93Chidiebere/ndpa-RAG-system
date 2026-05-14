import os
import json
import re
from pdf2image import convert_from_path
import pytesseract

def clean_text(text):
    text = re.sub(r'\s+', " ", text)
    text = re.sub(r'\n+', "\n", text)
    text = text.strip()
    return text

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def extract_text_with_ocr(pdf_path):
    print("Extracting text via OCR... this might take a few minutes depending on your system.")
    pages = convert_from_path(pdf_path)
    full_text = ""
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        full_text += text
        print(f"Processed page {i+1}/{len(pages)}")
    return full_text

if __name__ == "__main__":
    pdf_path = "Nigeria-Data-Protection-Act-2023.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        exit(1)
        
    print("Starting data preparation...")
    raw_text = extract_text_with_ocr(pdf_path)
    
    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)
    
    print("Chunking text...")
    chunks = chunk_text(cleaned_text)
    
    print(f"Total chunks created: {len(chunks)}")
    
    output_file = "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully saved {len(chunks)} chunks to {output_file}")
