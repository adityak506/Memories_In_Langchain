"""
03_buffer_window_memory.py (Modern LangChain Version)
-----------------------------------------------------
Concept: Chatbot with Limited (Windowed) Memory

Purpose:
Demonstrate how to limit chat history to the last N messages using
LangChain’s modern RunnableWithMessageHistory API.

LangChain Concept: RunnableWithMessageHistory (custom window truncation)
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
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

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


# ==========================================
# 2️⃣ Define Prompt + Base Chain
# ==========================================
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\n"
    "Conversation so far (last few turns): {history}\n"
    "User: {input}\n"
    "Assistant:"
)

base_chain = prompt | llm | StrOutputParser()

# ==========================================
# 3️⃣ Configure Custom Windowed Memory
# ==========================================
WINDOW_SIZE = 4  # number of messages to retain (user + assistant pairs)
store = {}

def get_session_history(session_id: str):
    """Returns or creates a message history with limited window."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    history = store[session_id]

    # Keep only the last N messages
    if len(history.messages) > WINDOW_SIZE:
        history.messages = history.messages[-WINDOW_SIZE:] #slice to keep only the last WINDOW_SIZE messages
    return history

# ==========================================
# 4️⃣ Wrap Chain with Message History
# ==========================================
chain_with_lastNMessages_memory_chat = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ==========================================
# 5️⃣ Interactive CLI Chat
# ==========================================
print("\n💬 CUSTOMER SUPPORT CHAT — WINDOWED MEMORY (Modern API)")
print(f"(Keeps only the last {WINDOW_SIZE} messages)")
print("Type 'exit' or 'quit' to end.\n")
print("----------------------------------------")

session_id = "window_session"

while True:
    user_input = input("👤 You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Chat ended. Thanks for testing!")
        break

    if not user_input:
        continue

    response = chain_with_lastNMessages_memory_chat.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    print(f"🤖 Assistant: {response}\n")

    # Optional: show live window size info
    history = get_session_history(session_id)
    print(f"🧠 Memory retains {len(history.messages)} of {WINDOW_SIZE} max messages\n")
    
# ==========================================
# 6️⃣ Final Memory State, when we exit the chat then to see the history of the chat we 
# can see the final memory state of the chat. here we see last 4 messages of the chat 
# as we have set the window size to 4.
# ==========================================
print("----------------------------------------")
print(f"🧠 Final Memory (last {WINDOW_SIZE} exchanges):\n")

for msg in get_session_history(session_id).messages:
    role = "USER" if msg.type == "human" else "ASSISTANT"
    print(f"{role}: {msg.content}")

print("""
----------------------------------------
📘 Key Takeaways:
1️⃣ RunnableWithMessageHistory can easily simulate windowed memory.
2️⃣ Trimming history manually avoids deprecated classes.
3️⃣ Ideal for fast, short-context assistants (like live chatbots).
""")
