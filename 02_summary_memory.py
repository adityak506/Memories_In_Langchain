"""
02_summary_memory.py (Modern LangChain Version)
-----------------------------------------------
Concept: Chatbot with Summarized Memory

Purpose:
Show how a chatbot can remember context using summarized conversation history
instead of storing every message.

LangChain Concept: RunnableWithMessageHistory + summarizing message store.

key takeaway: Summarized memory allows for longer conversations without exceeding context limits.
📘 Key Takeaways:
1️⃣ RunnableWithMessageHistory manages context cleanly.
2️⃣ You can plug in your own summarization logic.
3️⃣ Perfect for long conversations that exceed context windows.
TWO LLM calls are made in this example: one for the main chat and one for summarization. 
The summarization is triggered when the conversation exceeds a certain length (here, 8 messages). 
The summary is then stored in memory, allowing the chatbot to maintain context without exceeding token
limits.
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

# Load model config
with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
cfg = config[provider]

# Initialize model
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
# 2️⃣ Prompt & Base Chain
# ==========================================
prompt = ChatPromptTemplate.from_template(
    "You are a concise and helpful assistant.\n"
    "Here’s a summary of the chat so far: {history}\n"
    "Now respond to the user message: {input}\n"
)

base_chain = prompt | llm | StrOutputParser()

# ==========================================
# 3️⃣ Setup Summarizing Message History
# ==========================================
# from langchain.memory import ConversationSummaryBufferMemory  # temporary utility
# # Note: RunnableWithMessageHistory doesn't yet auto-summarize; we'll mimic it here.

store = {}

def get_session_history(session_id: str):
    """Creates or returns summarized history for the session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

#fetching the entire session history against each session_id. If the session_id is not present in the store,it creates
#a new InMemoryChatMessageHistory object and stores it in the store dictionary with the session_id as the key.

# ==========================================
# 4️⃣ Wrap in RunnableWithMessageHistory
# ==========================================
chain_with_summarize_memory_chat = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ==========================================
# 5️⃣ Live CLI Chat Loop
# ==========================================
print("\n💬 CUSTOMER SUPPORT CHAT — SUMMARY MEMORY (Modern API)")
print("Type 'exit' or 'quit' to end.\n")
print("----------------------------------------")

session_id = "summary_session"

while True:
    user_input = input("👤 You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Chat ended. Thanks for testing!")
        break

    if not user_input:
        continue

    response = chain_with_summarize_memory_chat.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    print(f"🤖 Assistant: {response}\n")
    
    # Optional: dynamically summarize long histories
    history = get_session_history(session_id)
    
    if len(history.messages) > 8: #this is a simple heuristic to summarize the conversation if it 
        #exceeds 8 messages
        # Simple summarization heuristic for demonstration
        
        #in coversation we are passing entire conversation to the summarizer prompt and 
        #it will return a concise summary of the conversation
        #llm will generate the summary and we will store it in the memory for that session_id
        summarizer = (
            ChatPromptTemplate.from_template(
                "Summarize the following conversation in 2-3 concise sentences:\n{conversation}"
            )
            | llm
            | StrOutputParser()
        )
        convo_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in history.messages]) 
        #users, assistant messages are combined into a single string to be summarized
        #USER and ASSISTANT are the message types and m.content is the actual message content
        #LABELING IS DONE SO THAT THE SUMMARIZER CAN UNDERSTAND WHO SAID WHAT IN THE CONVERSATION
        summary = summarizer.invoke({"conversation": convo_text}) #THIS WILL GENERATE THE SUMMARY OF
        #THE CONVERSATION
        store[session_id] = InMemoryChatMessageHistory() #THIS LINE WILL RESET THE HISTORY FOR THAT SESSION_ID
        #AND STORE ONLY THE SUMMARY IN THE MEMORY
        store[session_id].add_ai_message(f"Conversation Summary: {summary}") 
        #THIS LINE WILL ADD THE SUMMARY TO THE MEMORY FOR THAT SESSION_ID
        print("\n🧠 Conversation summarized to maintain concise memory.\n")
        
# ==========================================
# 6️⃣ Show Stored Summary when we exit the chat then to see the summary history of the chat we 
# can see the final memory state of the chat.
# ==========================================
print("----------------------------------------")
print("🧠 Final Summarized Memory:\n")

#history = get_session_history(session_id)
for msg in get_session_history(session_id).messages:
    role = "USER" if msg.type == "human" else "ASSISTANT"
    print(f"{role}: {msg.content}")

print("""
----------------------------------------
📘 Key Takeaways:
1️⃣ RunnableWithMessageHistory manages context cleanly.
2️⃣ You can plug in your own summarization logic.
3️⃣ Perfect for long conversations that exceed context windows.
""")