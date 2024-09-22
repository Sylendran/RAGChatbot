import openai
import streamlit as st
from utils.pdf_utils import extract_text_from_pdf, split_text  # Import PDF utility function
from utils.huggingface_utils import upload_to_huggingface 
from utils.embedding_utils import create_embeddings
from utils.huggingface_download_embed_utils import download_and_embed_pdfs_from_huggingface
from utils.pinecone_utils import init_pinecone, init_vector_store
import tempfile


st.title("RAG Agentforce")
st.header("Experimental project with OpenAI LLM + RAG using Huggingface + Pinecone vectorstore")

# Initialize session state for files and their content
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = {}

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = {}

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Set Pinecone & Hugging Face API tokens
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_index_name = st.secrets["PINECONE_INDEX_NAME"]  # Use the new index-specific setup
hf_token = st.secrets["HUGGINGFACE_API_KEY"]
repo_id = "Sylendran/RChatbot"
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize Pinecone and vector store
pinecone_index = init_pinecone(pinecone_api_key, pinecone_index_name)
vector_store, embedding_model = init_vector_store(pinecone_index)


# Download and embed PDFs from Hugging Face repo when app starts
if 'data_loaded' not in st.session_state:
    st.write("Downloading and embedding files from Hugging Face repo...")

    # Download and embed PDFs
    st.session_state['embeddings'], downloaded_files = download_and_embed_pdfs_from_huggingface(repo_id, hf_token)

    # Add the downloaded files to the session state
    for file_name, file_path in downloaded_files.items():
        st.session_state['uploaded_files'][file_name] = file_path

    # Add the embeddings to Pinecone
    for file_name, embedding in st.session_state['embeddings'].items():
        # Extract text content from the uploaded file
        pdf_text = extract_text_from_pdf(st.session_state['uploaded_files'][file_name])

        # Add the text and metadata to Pinecone (text as the document, and metadata as file name)
        #vector_store.add_texts(texts = [pdf_text], metadatas=[{"file_name": file_name}])
        # Split the text into smaller chunks to avoid exceeding Pinecone's size limit
        text_chunks = split_text(pdf_text, chunk_size=1000)  # Adjust chunk size based on your needs

        #  Add each chunk to Pinecone with file metadata (e.g., file_name + chunk index)
        for idx, chunk in enumerate(text_chunks):
            vector_store.add_texts(texts=[chunk], metadatas=[{"file_name": f"{file_name}_part_{idx}"}])

    st.session_state['data_loaded'] = True
    st.sidebar.write("Downloaded and embedded PDFs from Hugging Face repository.")
    #st.write(f"Total embeddings loaded: {len(st.session_state['embeddings'])}")

# Sidebar for uploading PDF files
st.sidebar.header("Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Handle uploaded files and add them to Hugging Face
if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state['uploaded_files']:
            # Create a temporary file to store the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.getbuffer())
                temp_file_path = temp_file.name
            
            st.session_state['uploaded_files'][file.name] = file
            st.sidebar.write(f"Uploaded: {file.name}")

            pdf_text = extract_text_from_pdf(file)

            # Generate embedding for the PDF text
            embeddings = create_embeddings([pdf_text])
            st.session_state['embeddings'][file.name] = embeddings

            # Add the new embeddings to Pinecone
            #vector_store.add_texts(texts = [pdf_text], metadatas=[{"file_name": file.name}])
            # Split the text into smaller chunks to avoid exceeding Pinecone's size limit
            text_chunks = split_text(pdf_text, chunk_size=1000)  # Adjust chunk size based on your needs

            # Add each chunk to Pinecone with file metadata (e.g., file_name + chunk index)
            for idx, chunk in enumerate(text_chunks):
                vector_store.add_texts(texts=[chunk], metadatas=[{"file_name": f"{file.name}_part_{idx}"}])

            # Upload to Hugging Face
            upload_to_huggingface(file, repo_id, hf_token)

# Display the uploaded files and provide an option to delete each file
if st.session_state['uploaded_files']:
    st.sidebar.write("Uploaded Files:")
    files_to_delete = []
    for file_name in st.session_state['uploaded_files'].keys():
        col1, col2 = st.sidebar.columns([3, 1])  # Create two columns for file display and delete button
        col1.write(file_name)
        if col2.button(f"Delete {file_name}", key=file_name):
            files_to_delete.append(file_name)

    # Remove selected files from session state
    for file_name in files_to_delete:
        del st.session_state['uploaded_files'][file_name]
        st.sidebar.write(f"Deleted: {file_name}")


# RAG-Based Answering
def rag_based_answer(query, threshold=0.10):
    query_embedding = embedding_model.embed_query(query)

   # Search Pinecone for top documents and their similarity scores
    docs_and_scores = vector_store.similarity_search_with_score(query, k=5)  # Retrieve scores as well
    #st.write(f"Retrieved documents and scores: {docs_and_scores}")
    
    # Check if we got any results
    if not docs_and_scores:
        st.write("No relevant documents found. Falling back to GPT search.")
        return open_search_answer_stream(query)
    
    # Unpack the top document and its score
    top_doc, top_score = docs_and_scores[0]  # Unpack only the first result (doc, score)
    
    # If the top score is below the threshold, fall back to open search
    if top_score < threshold:
        st.write(f"Top similarity score ({top_score}) below threshold ({threshold}). Falling back to GPT search.")
        return open_search_answer_stream(query)
    
    # Extract the page_content from the document objects
    retrieved_texts = [doc.page_content for doc, score in docs_and_scores]  # Don't unpack score
    
    #if not retrieved_texts:
    #    st.write("No relevant high-score documents found. Falling back to GPT search.")
    #    return open_search_answer_stream(query)

    document_context = "\n\n".join(retrieved_texts)

    # LLM call
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."},
        {"role": "user", "content": f"Based on the following documents, answer the question as briefly as possible:\n\n{document_context}\n\nQuestion: {query}\nAnswer:"}
    ]

    response = openai.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
        max_tokens=300,
        stream=True
    )

    # Stream the response
    full_response = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
            yield full_response


# Function to perform open-ended search with streaming using GPT model
def open_search_answer_stream(question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers general knowledge questions."},
        {"role": "user", "content": f"Answer the following question in the shortest and funniest way possible:\n\nQuestion: {question}\nAnswer:"}
    ]

    # Call OpenAI Chat API for streaming response
    response = openai.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
        max_tokens=300,
        stream=True  # Enable streaming
    )

    # Return chunks as they come
    full_response = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
        yield full_response  # Yield partial responses
    
# Display previous messages in the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input and process the query
prompt = st.chat_input("How can I assist you today?")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Use RAG-based search with FAISS and GPT to generate the response
        if st.session_state['embeddings']:  
            for partial_response in rag_based_answer(prompt):
                full_response = partial_response
                message_placeholder.markdown(full_response + "...")
        else:
            # Fallback to open-ended search with GPT
            for partial_response in open_search_answer_stream(prompt):
                full_response = partial_response
                message_placeholder.markdown(full_response + "...")

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
