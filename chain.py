import os
import streamlit as st

# -----------------------------
# LOAD SECRETS
# -----------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
LANGSMITH_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ds_mentor_app"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# -----------------------------
# IMPORTS
# -----------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# EMBEDDINGS
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# VECTOR STORE (FAISS - SAFE)
# -----------------------------
@st.cache_resource
def get_vectorstore():
    return FAISS.from_texts(["init"], embedding)

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# MEMORY
# -----------------------------
memory = InMemoryChatMessageHistory()

# -----------------------------
# PROMPTS
# -----------------------------
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Data Science Mentor.

Answer ONLY from the provided context.

Keep answers short.

Context:
{context}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

generic_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful Data Science Mentor.

- Answer clearly
- Keep it short
- End with 1 pro tip
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# -----------------------------
# MODEL
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.5
)

parser = StrOutputParser()

# -----------------------------
# ROUTER
# -----------------------------
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Classify the user question into:

GENERIC or RAG

Respond ONLY with one word.
"""),
    ("human", "{question}")
])

router_chain = router_prompt | llm | parser

def route_question(query: str) -> str:
    return router_chain.invoke({"question": query}).strip().upper()

# -----------------------------
# PDF → VECTORSTORE
# -----------------------------
def load_pdf_to_chroma(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    vectorstore.add_texts([doc.page_content for doc in docs])

    return "PDF uploaded and indexed!"

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def get_response(user_input: str) -> str:

    route = route_question(user_input)

    if route == "GENERIC":
        chain = generic_prompt | llm | parser
        response = chain.invoke({
            "question": user_input,
            "chat_history": memory.messages
        })

    else:
        docs = retriever.invoke(user_input)

        if not docs:
            chain = generic_prompt | llm | parser
            response = chain.invoke({
                "question": user_input,
                "chat_history": memory.messages
            })
        else:
            context = "\n\n".join([doc.page_content for doc in docs])

            chain = rag_prompt | llm | parser
            response = chain.invoke({
                "question": user_input,
                "chat_history": memory.messages,
                "context": context
            })

    memory.add_message(HumanMessage(content=user_input))
    memory.add_message(AIMessage(content=response))

    return response