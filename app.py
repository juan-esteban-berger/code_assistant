import streamlit as st
import random
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


@st.cache_resource
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("/home/juanesh/Models/deepseek-coder-1.3b-base", local_files_only=True)
    return tokenizer

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("/home/juanesh/Models/deepseek-coder-1.3b-base", local_files_only=True)
    return model

tokenizer = load_tokenizer()
model = load_model()

token_val = st.number_input("Max Number of New Tokens", value=40)
#language_val = st.text_input('Language', 'python')
#st.write("https://github.com/react-syntax-highlighter/react-syntax-highlighter/blob/master/AVAILABLE_LANGUAGES_PRISM.MD")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.code(message["content"])

if prompt := st.chat_input("Pase code here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.code(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=token_val)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.code(f'{answer}')
    st.session_state.messages.append({"role": "assistant", "content": answer})
