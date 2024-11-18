import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MARITACA_API_KEY = os.getenv("MARITACA_API_KEY")