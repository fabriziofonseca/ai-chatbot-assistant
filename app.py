import streamlit as st
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI

# Load env vars and DeepSeek
load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

st.set_page_config(page_title="Radiant Medspa AI", page_icon="💬")
st.title("Hello! I'm your AI front desk Assistant! How I can help you today?")

# Extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Vectorstore caching
@st.cache_resource(show_spinner="Indexing and embedding PDF...")
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)
    chunks = splitter.create_documents([text])
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local("faiss_index")
    return vectorstore

# Restore from disk if exists
if os.path.exists("faiss_index/index.faiss") and os.path.exists("faiss_index/index.pkl"):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.load_local(
        "faiss_index",
        embedder,
        allow_dangerous_deserialization=True
)

else:
    st.session_state.vectorstore = None

# Init memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
pdf_file = st.file_uploader("Upload your files please", type="pdf")
if pdf_file and st.session_state.vectorstore is None:
    text = extract_text_from_pdf(pdf_file)
    st.session_state.vectorstore = create_vectorstore(text)

# Input
prompt = st.chat_input("Ask a question about services, prices, treatments...")

# Process query
if prompt and st.session_state.vectorstore:
    # Save user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Display full past chat (from oldest to newest)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Prepare messages
    docs = st.session_state.vectorstore.similarity_search(prompt, k=2)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = [{"role": "system", "content": "You are a helpful assistant that works for a medspa, your job is to provide clients information based ONLY in the files provided. You will look forward to book an appointment with the prospect and also keep the information short but charming and charismatic like a front desk assistant."}]
    messages += st.session_state.chat_history
    messages.append({"role": "user", "content": f"{prompt}\n\nContext:\n{context}"})

    # Stream assistant reply
    response_text = ""
    with st.chat_message("assistant"):
        response_box = st.empty()
        response_stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", "")
            response_text += content
            response_box.markdown(response_text)

    # Save assistant reply to history
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
