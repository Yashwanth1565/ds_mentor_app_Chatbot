# -----------------------------
# LANGSMITH SETUP (FIRST!)
# -----------------------------
import os

print("🚀 Initializing LangSmith...")

with open("langsmithapi.txt", "r") as f:
    LANGSMITH_API_KEY = f.read().strip()

os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ds_mentor_app"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

print("✅ LangSmith Enabled")

# -----------------------------
# IMPORTS
# -----------------------------
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# API KEYS
# -----------------------------
with open("groqapi.txt", "r") as f:
    GROQ_API_KEY = f.read().strip()

# -----------------------------
# EMBEDDINGS
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# VECTOR STORE
# -----------------------------
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

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

If context says "Placement POC is Shaheer sir"
then answer: "Shaheer sir is the Placement POC."

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
Classify the user question into one of two categories:

1. GENERIC → greetings, basic concepts
2. RAG → needs document lookup

Respond ONLY with:
GENERIC or RAG
"""),
    ("human", "{question}")
])

router_chain = router_prompt | llm | parser

def route_question(query: str) -> str:
    return router_chain.invoke({"question": query}).strip().upper()

# -----------------------------
# PDF LOADER
# -----------------------------
def load_pdf_to_chroma(file_path: str):

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    vectorstore.add_documents(docs)
    vectorstore.persist()

    return "✅ PDF stored successfully!"

# -----------------------------
# MAIN RESPONSE FUNCTION
# -----------------------------
def get_response(user_input: str) -> str:

    print("📩 User:", user_input)

    route = route_question(user_input)
    print("🧠 Route:", route)

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

    print("🤖 Response:", response)

    return response