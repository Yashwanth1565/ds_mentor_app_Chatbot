import streamlit as st
from chain import get_response, load_pdf_to_chroma

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="DS Mentor", layout="wide")

st.title("📊 DS Mentor Chatbot")

# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# SIDEBAR (FIXED)
# -----------------------------
with st.sidebar:
    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        st.success(load_pdf_to_chroma("temp.pdf"))

# -----------------------------
# CHAT DISPLAY
# -----------------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        response = get_response(user_input)
        st.write(response)

    st.session_state.chat_history.append(("assistant", response))