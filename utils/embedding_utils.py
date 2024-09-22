from langchain.embeddings import HuggingFaceEmbeddings

def create_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create embeddings for a list of texts using HuggingFaceEmbeddings via Langchain.
    """
    model = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = model.embed_documents(texts)
    return embeddings
