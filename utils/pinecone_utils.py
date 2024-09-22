from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
import os

def init_pinecone(api_key, index_name):
    """
    Initialize Pinecone client and ensure the index exists.
    """
    pc = Pinecone(api_key=api_key)

    # Check if the index exists, if not, create it
    if index_name not in [i.name for i in pc.list_indexes()]:
        # Create the index with the appropriate dimensions (e.g., 1536 for OpenAI embeddings)
        pc.create_index(
            name=index_name,
            dimension=1536,  # Ensure this matches your embedding dimensions
            metric='cosine',  # Metric can be 'cosine', 'euclidean', etc.
            spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust region and cloud if necessary
        )
    
    return pc.Index(index_name)

def init_vector_store(pinecone_index):
    """
    Initialize the LangChain vector store with Pinecone.
    """
    model_name="all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    #embeddings = model.embed_documents(texts)
    #embedding_model = OpenAIEmbeddings()
    vector_store = LangchainPinecone(pinecone_index, embedding_model.embed_query, "text")
    return vector_store, embedding_model
