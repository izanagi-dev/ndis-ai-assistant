import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. SETUP
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="NDIS Assistant", page_icon="⚖️")
st.header("⚖️ NDIS AI Assistant")

# Sidebar
with st.sidebar:
    st.title("Navigation")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

@st.cache_resource
def get_vectorstore():
    loader = TextLoader("knowledge_base.txt", encoding="utf-8")
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)
    return FAISS.from_documents(docs, embeddings)

# 2. INITIALIZE
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.1)

# 3. CHAIN DEFINITION
template = """You are an expert NDIS advisor. Answer based ONLY on the context.
Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

rag_chain = (
    {
        "context": itemgetter("question") | retriever, 
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 4. CHAT HISTORY DISPLAY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. EXECUTION
if user_input := st.chat_input("Ask about NDIS eligibility..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    docs = retriever.invoke(user_input)
    
    with st.spinner("Consulting NDIS Guidelines..."):
        response = rag_chain.invoke({
            "question": user_input, 
            "chat_history": st.session_state.chat_history
        })
    
    with st.chat_message("assistant"):
        st.markdown(response)
        with st.expander("Sources"):
            for doc in docs:
                st.caption(doc.page_content)
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})