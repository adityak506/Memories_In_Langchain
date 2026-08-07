"""
05_streamlit_memory.py
---------------------------------------------
Streamlit UI for LangChain memory examples.

Usage:
    streamlit run 05_streamlit_memory.py
"""

import json
import os
import uuid
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from tiktoken import get_encoding

# ------------------------------------------
# 1️⃣ Setup
# ------------------------------------------
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
cfg = config[provider]


def make_llm():
    if provider == "openai":
        return ChatOpenAI(
            model=cfg.get("model"),
            temperature=cfg.get("temperature", 0.6),
            max_tokens=cfg.get("max_tokens", 250),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    return ChatGoogleGenerativeAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.6),
        max_output_tokens=cfg.get("max_output_tokens", 250),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


llm = make_llm()
parser = StrOutputParser()

# ------------------------------------------
# 2️⃣ Prompt templates + chains
# ------------------------------------------
basic_prompt = ChatPromptTemplate.from_template(
    "You are a friendly customer support assistant.\n"
    "Previous messages: {history}\n"
    "User: {input}\n"
    "Assistant:"
)

summary_prompt = ChatPromptTemplate.from_template(
    "You are a concise and helpful assistant.\n"
    "Here’s a summary of the chat so far: {history}\n"
    "Now respond to the user message: {input}\n"
)

window_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\n"
    "Conversation so far (last few turns): {history}\n"
    "User: {input}\n"
    "Assistant:"
)

token_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\n"
    "Conversation so far (trimmed to fit token limit): {history}\n"
    "User: {input}\n"
    "Assistant:"
)

base_chain_basic = basic_prompt | llm | parser
base_chain_summary = summary_prompt | llm | parser
base_chain_window = window_prompt | llm | parser
base_chain_token = token_prompt | llm | parser

# ------------------------------------------
# 3️⃣ Memory helpers
# ------------------------------------------
WINDOW_SIZE = 4
TOKEN_LIMIT = 250
encoding = get_encoding("cl100k_base")

if "chat_store" not in st.session_state:
    st.session_state["chat_store"] = {}


def num_tokens_from_messages(messages: List[str]) -> int:
    return sum(len(encoding.encode(m)) for m in messages)


def get_basic_history(session_id: str):
    store = st.session_state["chat_store"]
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def get_summary_history(session_id: str):
    store = st.session_state["chat_store"]
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def get_window_history(session_id: str):
    store = st.session_state["chat_store"]
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    history = store[session_id]
    if len(history.messages) > WINDOW_SIZE:
        history.messages = history.messages[-WINDOW_SIZE:]
    return history


def get_token_history(session_id: str):
    store = st.session_state["chat_store"]
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    history = store[session_id]
    total_tokens = num_tokens_from_messages([m.content for m in history.messages])
    while total_tokens > TOKEN_LIMIT and len(history.messages) > 2:
        history.messages = history.messages[2:]
        total_tokens = num_tokens_from_messages([m.content for m in history.messages])

    return history

# ------------------------------------------
# 4️⃣ Runnable chains
# ------------------------------------------
chain_basic = RunnableWithMessageHistory(
    base_chain_basic,
    get_basic_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_summary = RunnableWithMessageHistory(
    base_chain_summary,
    get_summary_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_window = RunnableWithMessageHistory(
    base_chain_window,
    get_window_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_token = RunnableWithMessageHistory(
    base_chain_token,
    get_token_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ------------------------------------------
# 5️⃣ Streamlit UI
# ------------------------------------------
st.set_page_config(page_title="LangChain Memory Chat", layout="wide")
st.title("LangChain Memory Chat")

st.sidebar.header("Memory Strategy")
memory_mode = st.sidebar.selectbox(
    "Choose a memory strategy:",
    ["Basic", "Summarized", "Windowed", "Token-Limited"],
)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

history_session_id = f"{st.session_state.session_id}-{memory_mode}"

if st.sidebar.button("Clear chat history"):
    st.session_state["chat_store"].pop(history_session_id, None)
    st.session_state.pop("user_input", None)

st.sidebar.markdown("---")
st.sidebar.markdown("### Notes")
if memory_mode == "Basic":
    st.sidebar.write("Stores all messages in memory.")
elif memory_mode == "Summarized":
    st.sidebar.write("Summarizes long conversations to keep memory compact.")
elif memory_mode == "Windowed":
    st.sidebar.write(f"Keeps only the last {WINDOW_SIZE} messages.")
else:
    st.sidebar.write(f"Keeps memory within ~{TOKEN_LIMIT} tokens.")

chain_map = {
    "Basic": chain_basic,
    "Summarized": chain_summary,
    "Windowed": chain_window,
    "Token-Limited": chain_token,
}

history_getter_map = {
    "Basic": get_basic_history,
    "Summarized": get_summary_history,
    "Windowed": get_window_history,
    "Token-Limited": get_token_history,
}

chain = chain_map[memory_mode]
history = history_getter_map[memory_mode](history_session_id)

st.subheader("Conversation")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message")
    submit = st.form_submit_button("Send")

if submit and user_input:
    user_message = user_input.strip()
    if user_message:
        response = chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": history_session_id}},
        )

        history = history_getter_map[memory_mode](history_session_id)

        if memory_mode == "Summarized" and len(history.messages) > 8:
            summarizer = (
                ChatPromptTemplate.from_template(
                    "Summarize the following conversation in 2-3 concise sentences:\n{conversation}"
                )
                | llm
                | parser
            )
            convo_text = "\n".join(
                [f"{m.type.upper()}: {m.content}" for m in history.messages]
            )
            summary = summarizer.invoke({"conversation": convo_text})
            st.session_state["chat_store"][history_session_id] = InMemoryChatMessageHistory()
            st.session_state["chat_store"][history_session_id].add_ai_message(
                f"Conversation Summary: {summary}"
            )
            st.success("Conversation summarized to maintain concise memory.")
            history = get_summary_history(history_session_id)

st.write("---")
for msg in history.messages:
    role = "USER" if msg.type == "human" else "ASSISTANT"
    st.markdown(f"**{role}:** {msg.content}")

if memory_mode == "Token-Limited":
    total_tokens = num_tokens_from_messages([m.content for m in history.messages])
    st.sidebar.write(f"Approx. tokens stored in memory: {total_tokens}")

st.markdown("---")
st.caption(
    "This demo uses the modern LangChain RunnableWithMessageHistory API with a Streamlit UI."
)
