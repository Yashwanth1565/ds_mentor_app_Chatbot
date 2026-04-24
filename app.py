import streamlit as st
from chain import get_response, load_pdf_to_chroma

# -----------------------------
# PAGE CONFIG (FIX SIDEBAR BUG)
# -----------------------------
st.set_page_config(
    page_title="DS Mentor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💬 DS Mentor App")

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# SIDEBAR (FIXED)
# -----------------------------
with st.sidebar:
    st.title("📜 Chat History")

    if not st.session_state.messages:
        st.write("No chats yet...")
    else:
        for i, msg in enumerate(st.session_state.messages):
            role = "🧑" if msg["role"] == "user" else "🤖"
            st.write(f"{role} {msg['content'][:40]}...")

# -----------------------------
# PDF UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing PDF..."):
        result = load_pdf_to_chroma("temp.pdf")
        st.success(result)

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Get response
    with st.spinner("Thinking..."):
        response = get_response(user_input)

    # Show assistant response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })