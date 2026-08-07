"""
01_basic_memory.py (Modern LangChain Version)
---------------------------------------------
Concept: Interactive Chatbot WITH Memory (Runnable API)

Purpose:
Demonstrate a live terminal chatbot that remembers previous conversation turns,
using LangChain’s new Runnable + MessageHistory system.

LangChain Concept: RunnableWithMessageHistory
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
#InMemoryChatMessageHistory is a class that stores chat messages in memory.
#It is used to keep track of the conversation history between the user and the chatbot.
from langchain_core.runnables.history import RunnableWithMessageHistory
#RunnableWithMessageHistory is a class that allows you to create a runnable that can maintain
#a message history.

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

parser = StrOutputParser()

# ==========================================
# 2️⃣ Define Prompt + Chain
# ==========================================
prompt = ChatPromptTemplate.from_template(
    "You are a friendly customer support assistant.\n"
    "Previous messages: {history}\n"
    "User: {input}\n"
    "Assistant:"
)

base_chain = prompt | llm | parser


# ==========================================
# 3️⃣ Configure Memory Store
# ==========================================
# Create a simple in-memory store for sessions since we don't have a database in this example.
store = {} #for storing the chat history as per session_id.
#It is a dictionary where the key is the session_id and the value is the InMemoryChatMessageHistory
#object for that session.

def get_session_history(session_id: str):
    """Returns or creates message history for a session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory() 
        #creates a new, empty chat history object that stores all messages (user and assistant) in memory (RAM).
        #so that the chatbot can remember the conversation history for that session and when we resume it remembers
        #the previous messages.
        
        #session_id is a unique identifier for the chat session, which allows the chatbot to keep
        #track of the conversation history for each user as the chat is stored using session_id
        #as the key in the store dictionary.
        #each and every session_id will have different chat history or different context.
    return store[session_id]

# Wrap the chain with memory
chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
# RunnableWithMessageHistory is a LangChain utility that wraps a "chain" (here, base_chain) 
# and automatically manages chat history (memory) for you.
# base_chain is your prompt + LLM + output parser pipeline.
# get_session_history is a function that, given a session ID, returns the correct 
# InMemoryChatMessageHistory object (from the store dictionary).
# input_messages_key="input" tells LangChain which key in your input dictionary contains the user’s message.
# history_messages_key="history" tells LangChain which key should be used to pass the chat history to the prompt.

# ==========================================
# 4️⃣ Interactive CLI Chat
# ==========================================
print("\n💬 CUSTOMER SUPPORT CHAT — WITH MODERN MEMORY")
print("Type 'exit' or 'quit' to end.\n")
print("----------------------------------------")

session_id = "demo_session" #for time being it is static but in real world it will be dynamic and 
#unique for each user session.

while True:
    user_input = input("👤 You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Chat ended. Thanks for testing!")
        break

    if not user_input:
        continue
    
    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ) 
    #this is the proper syntax to invoke the chain_with_memory object with the user input and session_id
    #when using RunnableWithMessageHistory, you need to pass the session_id in the config parameter
    #so that it can retrieve the correct chat history for that session.
    
    #invoke as the method only accept one argument so whenever we want to pass multiple arguments
    #we can use dictionary to pass multiple arguments.
    
    #apart from input the other one is a runnable config which is a dictionary that can be used to pass
    # additional configuration options to the runnable.
    print(f"🤖 Assistant: {response}\n")
    #now each and every response needs to be added back to the history so that the next time when we
    #invoke the chain_with_memory it will have the previous context
    
    # ==========================================
# 5️⃣ Display Stored Messages
# ==========================================
print("----------------------------------------")
print("🧠 Chat Memory Contents (Modern API):\n")

history = get_session_history(session_id)
for msg in history.messages:
    role = "USER" if msg.type == "human" else "ASSISTANT" #setting role is important for chatbot
    print(f"{role}: {msg.content}")

print("""
----------------------------------------
📘 Key Takeaways:
1️⃣ RunnableWithMessageHistory replaces ConversationChain.
2️⃣ Memory is session-based and flexible (InMemory, Redis, DB, etc.).
3️⃣ 100% compatible with LangChain 0.3+ — no deprecation warnings!
      
Summary: InMemoryChatMessageHistory is great for demos, testing, or small-scale use, but not for production or multi-user systems where persistence and scalability are needed.      
""")

'''
5 Display Stored Messages block is reading the saved chat history for the current session and printing each 
message with a label.

history = get_session_history(session_id)

Calls the function get_session_history(...) with the current session_id.
Returns the InMemoryChatMessageHistory object for that session.
If the session is new, it creates and stores a fresh history object.
for msg in history.messages:

Iterates over every saved message in that session’s history.
history.messages is a list of chat messages recorded so far.
role = "USER" if msg.type == "human" else "ASSISTANT"

Checks the message type.
If the message was sent by the human user, it sets role to "USER".
Otherwise, it sets role to "ASSISTANT".
print(f"{role}: {msg.content}")

Prints the message label and the actual text.
Example output:
USER: Hello
ASSISTANT: Hi there!
'''

