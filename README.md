# Intelligent PDF Chatbot

An AI-powered PDF chatbot that allows users to query and summarize content across one or more PDF documents. It uses Retrieval-Augmented Generation (RAG) with LangChain, Google Gemini Pro, and FAISS for intelligent document interaction.

## Features

- Ask questions from uploaded PDFs
- Powered by Google Gemini Pro LLM
- Handles multi-PDF input
- Uses FAISS for vector-based document retrieval
- Prompt engineering to ensure factual, grounded answers
- Contextual QA with LangChain chains

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Generative AI (Gemini)
- FAISS
- PyPDF2
- dotenv for environment variables

## Installation
# 1. Clone the Repository
git clone https://github.com/mugilsuga07/intelligent-pdf-chatbot.git
cd intelligent-pdf-chatbot

# 2. Create and Activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Required Dependencies
pip install -r requirements.txt

# 4. Create .env file and add your Gemini API Key
touch .env
echo "GOOGLE_API_KEY=your_api_key_here" >> .env

# 5. Run the Application
streamlit run chatpdf1.py

