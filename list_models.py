import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

models = genai.list_models()

for model in models:
    print(f"Name: {model.name}")
    print(f"Supports generation: {'generateContent' in model.supported_generation_methods}")
    print("-" * 50)
