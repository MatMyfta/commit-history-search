import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    REPO_URL = os.getenv("REPO_URL")
    REPO_DIR = os.getenv("REPO_DIR")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "./data/spring_boot_commits.json")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
