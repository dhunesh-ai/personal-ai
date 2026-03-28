import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 Your Personal Chatbot ")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Store
    db = FAISS.from_documents(docs, embeddings)
    st.session_state.db = db

    st.success("PDF processed successfully!")

# Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
query = st.chat_input("Ask something about your PDF...")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.db is None:
        st.warning("Please upload a PDF first!")
    else:
        # Retrieve
        docs = st.session_state.db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # LLM
        llm = Ollama(model="llama3:8b")

        prompt = f"""
        Answer ONLY from the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)