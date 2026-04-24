import os

# ✅ SET ENV FIRST
os.environ["LANGCHAIN_API_KEY"] = open("langsmithapi.txt").read().strip()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ds_mentor_app"

print("✅ ENV SET")

# ✅ IMPORT AFTER
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=open("groqapi.txt").read().strip()
)

print("🚀 Sending test request...")
response = llm.invoke("Hello from test")

print("🤖", response)