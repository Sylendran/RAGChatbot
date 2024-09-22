from huggingface_hub import HfApi
from utils.pdf_utils import extract_text_from_pdf
from utils.embedding_utils import create_embeddings
import requests
import os
import streamlit as st

def download_and_embed_pdfs_from_huggingface(repo_id, hf_token):
    """
    Download all PDF files from the Hugging Face repository,
    extract text, generate embeddings, and return both embeddings and file paths.
    
    :param repo_id: Hugging Face repository ID.
    :param hf_token: Hugging Face API token.
    :return: Dictionary of embeddings and a dictionary of file paths {filename: file_path}.
    """
    hf_api = HfApi()
    files = hf_api.list_repo_files(repo_id, repo_type="dataset", token=hf_token)
    embeddings_dict = {}
    downloaded_files = {}
    
    for file in files:
        if file.endswith(".pdf"):
            file_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file}"
            response = requests.get(file_url)
            if response.status_code == 200:
                # Save the PDF temporarily
                file_path = os.path.join("/tmp", file)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # Store the file path in downloaded_files (to add to session state later)
                downloaded_files[file] = file_path

                # Extract text from the PDF
                with open(file_path, "rb") as pdf_file:
                    pdf_text = extract_text_from_pdf(pdf_file)
                    #st.write(f"Extracted text from {file}: {pdf_text}")

                # Generate embedding for the text
                if pdf_text.strip():  # Only embed if text is not empty or just whitespace
                    embedding = create_embeddings([pdf_text])
                else:
                    st.write(f"Failed to extract text from {file}. Skipping embedding.")

                # Log the generated embedding using Streamlit
                #st.write(f"Generated embedding for {file}: {embedding}")
                
                # Store embedding in the dictionary
                embeddings_dict[file] = embedding

    return embeddings_dict, downloaded_files