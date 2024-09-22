# RAGChatbot
Experimental project with OpenAI LLM + RAG using Huggingface + Pinecone vectorstore

PDF AgentForce is an AI-powered chatbot designed to extract, index, and query information from uploaded PDF documents. It integrates advanced natural language processing techniques to provide document-centric responses, as well as fallback capabilities to open-ended GPT-based search when document context is insufficient.

**Key Features**
1. Pinecone Integration for Vector Search: Uses Pinecone to store and retrieve document embeddings, ensuring fast and efficient similarity search.
2. OpenAI GPT-4 Integration: Combines document-based answers with GPT-4 to provide general knowledge or fallback responses when documents don’t contain the relevant information.
3. Hugging Face Integration: Downloads and processes documents from a Hugging Face repository upon initialization.
4. Streamlit Interface: An intuitive UI that allows users to upload PDF documents, query them, and view responses interactively.

**Technology Stack**
- Pinecone: Used for storing and querying embeddings with high-performance vector search.
- OpenAI GPT-4: Powers natural language responses and fallback search capabilities.
- Streamlit: A lightweight and powerful framework used to build the app's user interface.
- Hugging Face Hub: Used for storing, downloading, and embedding PDF documents for document-based answers.
- Sentence Transformers: Utilized for generating embeddings for both the document content and user queries.

**How It Works**
1. Document Upload: Users can upload PDFs which are processed to extract text using the pdf_utils module.
2. Text Embedding: The extracted text is converted into embeddings using the Sentence Transformers model and stored in Pinecone.
3. Querying: When a query is made, the bot searches through the stored document embeddings in Pinecone to retrieve the most relevant information.
4. GPT Fallback: If the relevant context is not found in the document, the bot falls back to OpenAI GPT-4 to answer the query.
5. Response Streaming: Responses are streamed in real-time, offering a seamless user experience.


**Project Structure**
.
├── app_chatbot.py                     # The main application file for Streamlit
├── utils/
│   ├── pdf_utils.py             # Functions for PDF text extraction
│   ├── huggingface_utils.py     # Utility for uploading to Hugging Face
│   ├── embedding_utils.py       # Embedding generation using Sentence Transformers
│   ├── pinecone_utils.py        # Pinecone initialization and vector store setup
│   ├── huggingface_download_embed_utils.py  # Functions for downloading from Hugging Face
└── README.md                    # Project documentation


# install required libraries
pip install -r requirements.txt

Add the necessary environment variables:

PINECONE_API_KEY: Your Pinecone API key.
PINECONE_INDEX_NAME: The name of the Pinecone index.
OPENAI_API_KEY: Your OpenAI API key.
HUGGINGFACE_API_KEY: Your Hugging Face API key.

Run the application:
streamlit run app_chatbot.py

**Future Enhancements**
Advanced Document Parsing: Implement more robust parsing for a wider variety of document formats - Multimodal.
Improved Query Accuracy: Enhance the embedding and search mechanisms to deliver more precise answers.


Contributions
This is purely an experimental and hobby work! But I welcome contributions to improve this project!
