import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def retrieve_docs(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question)
    return docs

def generate_answer(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Use the following context to answer the question.
    If the answer is not in the context, say 'Answer not found in the provided context.'

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def main():
    st.set_page_config("Chat with PDF")
    st.title("Chat with PDF using Gemini")
    user_question = st.text_input("Ask a question about the uploaded PDFs")
    if user_question:
        with st.spinner("Thinking..."):
            docs = retrieve_docs(user_question)
            answer = generate_answer(user_question, docs)
            st.write("Reply:", answer)
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
                st.success("PDFs processed and indexed!")

if __name__ == "__main__":
    main()






