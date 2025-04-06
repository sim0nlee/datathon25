import streamlit as st
from openai import OpenAI
import json
import pandas as pd
import plotly.express as px

client = OpenAI(
    api_key="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")

# === CONFIG ===

st.set_page_config(page_title="LLM Chat with Data", layout="wide")
st.title("Supply Chain Chatbot")
st.subheader("Chat with your data using LLMs")
st.markdown(
    """
    This is a simple chatbot that uses OpenAI's GPT-4 model to answer questions about data, summaries, and charts.
    You can ask it anything related to your data and it will respond accordingly.
    """
)
st.sidebar.header("About")
st.sidebar.markdown(
    """
    This app is built using Streamlit and OpenAI's GPT-4 model.
    It allows you to interact with your data in a conversational manner.
    You can ask questions, get summaries, and even visualize data using charts.
    """
)
st.sidebar.markdown(
    """
    ### Instructions
    1. Type your question in the input box below.
    2. Press Enter to submit your question.
    3. The chatbot will respond with an answer or a chart.
    """
)
st.sidebar.markdown(
    """
    ### Example Questions
    - What is the summary of the data?
    - Show me a chart of sales over time.
    - What are the top 5 products by sales?
    """
)

# === SESSION STATE ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === HELPER ===
def ask_gpt(prompt):
    # st.session_state.messages.append({"role": "user", "content": prompt})
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    # )
    # message = response.choices[0].message.content
    # st.session_state.messages.append({"role": "assistant", "content": message})
    # return messages``
    # Mock response repeating the prompt for testing
    # In a real scenario, you would call the OpenAI API here
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = f"Mock response for: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    return response
    
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
user_input = st.chat_input("Ask me anything about data, summaries, or charts...")
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
