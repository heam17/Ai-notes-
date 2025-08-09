import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings

#OpenAI API
OPENAI_API_KEY = "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"
# 🧾 App Configuration
st.set_page_config(page_title="AI PDF Chatbot", layout="centered")
st.header("AI PDF Chatbot")

# 📤 Upload PDF file
with st.sidebar:
    st.title("📂 Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF to chat with its content", type="pdf")
    st.markdown("---")
    st.info("Supports basic question answering from PDFs.")

# 💬 Always display user input box
user_question = st.text_input("Ask a question about the PDF (after uploading)")

# ✅ When a file is uploaded
if uploaded_file is not None:
        # Extract PDF text
        pdf_reader = PdfReader(uploaded_file)
        full_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                full_text += text

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n", ".", "!", "?", ","],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = splitter.split_text(full_text)

        # Create LangChain Documents
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Generate embeddings
        embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, embeddings)

        # When user types a question
        if user_question:
            match = vector_store.similarity_search(user_question)

            #define llm
            llm = ChatOpenAI(
                openai_api_key = OPENAI_API_KEY,
                temperature= 0.8,
                max_tokens= 100,
                model_name = "gpt-3.5-turbo"
            )

            #output
            chain = load_qa_chain( llm , chain_type="stuff")
            response = chain.run(input_documents = match, question = user_question)
            st.write(response)