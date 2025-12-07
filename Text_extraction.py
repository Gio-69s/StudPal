from pypdf import PdfReader

def extract_text_from_a_PDF(file_path):
    try :
        reader= PdfReader(file_path)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text()
        return extracted_text
    except FileNotFoundError :
        return "File not found"
    
pdf_file = r"C:\Users\giova\OneDrive\Documents\GitHub\StudPal\datas\TD Maths leçon 01 LIMITES ET CONTINUITE.pdf"
text=extract_text_from_a_PDF(pdf_file)
print(text)
