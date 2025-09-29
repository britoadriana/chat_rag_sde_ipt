import streamlit as st
import uuid

def write_message(role, content, save=True):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if save:
        st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())
    return st.session_state['session_id']