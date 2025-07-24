import streamlit as st
import openai
import chromadb
from chromadb.utils import embedding_functions
import tempfile
import PyPDF2
import os
from dotenv import load_dotenv

# --- Load OpenAI API Key from .env ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit UI ---
st.title("RAG-based PDF Q&A with OpenAI")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file and openai_api_key:
    # --- Extract text from PDF ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    pdf_reader = PyPDF2.PdfReader(tmp_file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # --- Split text into chunks ---
    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    chunks = chunk_text(text)

    # --- Setup ChromaDB with OpenAI embeddings ---
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )
    """
    client = chromadb.Client()
    collection = client.create_collection(name="pdf_chunks_new", embedding_function=openai_ef)
    
    client = chromadb.Client()
    collection_name = "pdf_chunks_new"
    try:
        collection = client.create_collection(name=collection_name, embedding_function=openai_ef)
    except chromadb.errors.CollectionAlreadyExistsError:
        collection = client.get_collection(name=collection_name, embedding_function=openai_ef)
    """
        # ...existing code...
    
    client = chromadb.Client()
    collection_name = "pdf_chunks_new"
    # Check if collection exists and get or create accordingly
    if collection_name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(name=collection_name, embedding_function=openai_ef)
    else:
        collection = client.create_collection(name=collection_name, embedding_function=openai_ef)
    
    # ...existing code...


    # --- Add chunks to ChromaDB ---
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[str(i)]
        )

    st.success("PDF processed and indexed!")

    # --- Q&A ---
    query = st.text_input("Ask a question about the PDF:")
    if query:
        # Retrieve relevant chunks
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        retrieved_chunks = [doc for doc in results['documents'][0]]

        # Compose context for OpenAI
        context = "\n\n".join(retrieved_chunks)
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Call OpenAI Chat Completion
        openai.api_key = openai_api_key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for answering questions about a PDF document."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content.strip()
        st.markdown(f"**Answer:** {answer}")

   