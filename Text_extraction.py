import PyPDF2

# --- PDF Extraction ---
def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.
    Args:
        file_path (str): The path to the PDF file.
    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        return f"Error reading PDF file: {e}"
    return text if text else "No text found in the PDF."