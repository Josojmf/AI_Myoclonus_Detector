# app/utils.py
import os

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(filepath):
    extension = filepath.rsplit('.', 1)[1].lower()
    try:
        if extension == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif extension == 'pdf':
            import PyPDF2
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                return ''.join(page.extract_text() for page in pdf_reader.pages)
        elif extension == 'docx':
            import docx
            doc = docx.Document(filepath)
            return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        return f"Error reading file: {e}"
