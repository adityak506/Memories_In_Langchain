"""
00_no_memory.py
----------------
Concept: Interactive Chatbot WITHOUT Memory

Purpose:
Let learners chat live in the terminal and see that the model forgets previous inputs.

LangChain Concept: Direct LLM call (no memory)

How to Use:
$ python 00_no_memory.py

Type messages and press Enter.
Type 'exit' or 'quit' to end the chat.
"""

import os
import json
from dotenv import load_dotenv

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

# Load provider and configuration
with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
cfg = config[provider]

# Initialize LLM
if provider == "openai":
    llm = ChatOpenAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.6),
        max_tokens=cfg.get("max_tokens", 250),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.6),
        max_output_tokens=cfg.get("max_output_tokens", 250),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

parser = StrOutputParser()

# ==========================================
# 2️⃣ Define Prompt
# ==========================================
prompt = ChatPromptTemplate.from_template(
    "You are a friendly and professional customer support assistant.\n"
    "User message: {user_message}\n"
    "Respond politely and helpfully."
)

chain = prompt | llm | parser


# ==========================================
# 3️⃣ Live Chat Loop
# ==========================================
print("\n💬 CUSTOMER SUPPORT CHAT — NO MEMORY")
print("Type 'exit' or 'quit' to end.\n")
print("----------------------------------------")

while True:
    user_input = input("👤 You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Chat ended. Thanks for testing!")
        break

    if not user_input:
        continue

    # Invoke LLM chain for each message — no memory
    response = chain.invoke({"user_message": user_input})
    print(f"🤖 Assistant: {response}\n")