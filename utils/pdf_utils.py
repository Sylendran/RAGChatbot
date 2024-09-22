import PyPDF2

def extract_text_from_pdf(file):
    """
    Extract text content from a PDF file.
    """
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(reader.pages)):
        page_text = reader.pages[page_num].extract_text()
        if page_text:
            text += page_text
        if not text:
            text = "No extractable text"  # Log or handle this scenario
    return text