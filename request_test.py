import os
from dotenv import load_dotenv

load_dotenv()  # This line is MANDATORY to read your .env file

api_key = os.getenv("DEEPSEEK_API_KEY")
print(api_key)
