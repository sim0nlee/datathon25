import streamlit as st

import json
import pandas as pd
import plotly.express as px

import rag


# === GUI ===

st.set_page_config(page_title="LLM Chat with Data", layout="wide")
st.title("RAG(E Against the Machine Learning)")
st.subheader("Chat with your data using LLMs")

st.sidebar.markdown(
    """
    This is a simple chatbot interface developed by _Rage Against the Machine Learning™_ that allows you to ask questions about your data.
    The chatbot uses a Retrieval-Augmented Generation (RAG) approach to provide accurate answers.
    """
)

st.markdown(
    """
    ### Instructions
    - Ask any questions about the data.
    - The chatbot will provide answers based on the data and its knowledge.
    """
)


# === SESSION STATE ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === HELPER ===
def ask_gpt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    message = rag.answer(
        prompt,
        history=st.session_state.messages
    )
    st.session_state.messages.append({"role": "assistant", "content": message})
    return message
    
def try_parse_json(content):
    try:
        return json.loads(content)
    except Exception:
        return None

def render_content(reply):
    parsed = try_parse_json(reply)

    if parsed:
        if isinstance(parsed, list) and isinstance(parsed[0], dict):
            df = pd.DataFrame(parsed)
            st.dataframe(df)

            # Optional: Show a chart if there's numeric data
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) >= 2:
                st.subheader("📈 Chart")
                fig = px.bar(df, x=numeric_cols[0], y=numeric_cols[1])
                st.plotly_chart(fig)
        else:
            st.json(parsed)
    else:
        st.markdown(reply)


# === CHAT INPUT ===
user_input = st.chat_input("Ask me anything about your data")
if user_input:
    with st.spinner("Thinking..."):
        reply = ask_gpt(user_input)


# === CHAT DISPLAY ===
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            render_content(msg["content"])
