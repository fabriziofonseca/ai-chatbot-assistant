import streamlit as st
import fitz  # PyMuPDF
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load API Key securely (you can also use st.secrets["deepseek_api_key"])
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

st.title("📄💬 MedSpa Chatbot Powered by DeepSeek")

# PDF → Text
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Embed and store
@st.cache_resource(show_spinner="Indexing the PDF...")
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embedder = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")
    return FAISS.from_documents(docs, embedder)

# Ask DeepSeek
def ask_deepseek(query, context):
    prompt = f"""You are a helpful assistant for a medspa. Answer the user's question using ONLY the context below. If the info is not there, say "I’m not sure based on the uploaded info."

Context:
{context}

Question: {query}
Answer:"""

    res = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
    )
    return res.json()["choices"][0]["message"]["content"]

# Upload UI
uploaded_file = st.file_uploader("Upload MedSpa Service PDF", type="pdf")
if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    vectorstore = create_vectorstore(raw_text)
    st.success("PDF successfully parsed and indexed!")

    query = st.text_input("Ask about services, prices, or promotions:")
    if query:
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n\n".join([d.page_content for d in docs])
        with st.spinner("Thinking..."):
            response = ask_deepseek(query, context)
        st.markdown(f"🤖 **DeepSeek:** {response}")
