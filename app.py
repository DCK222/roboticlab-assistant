import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Pagina ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoboticLab Assistant",
    page_icon="🤖",
    layout="centered",
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .chat-user   { background:#1e3a5f; border-radius:12px; padding:10px 14px; margin:4px 0; }
    .chat-bot    { background:#0f2027; border-radius:12px; padding:10px 14px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("# 🤖")
with col2:
    st.title("RoboticLab Assistant")
    st.caption("Asistente inteligente sobre actividades de RoboticLab")

st.divider()

# ── API Key ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

if not GROQ_API_KEY:
    st.warning("⚠️ Ingresa tu GROQ_API_KEY en los secretos de Streamlit o en `.env`")
    GROQ_API_KEY = st.text_input("GROQ API Key", type="password", placeholder="gsk_...")
    if not GROQ_API_KEY:
        st.stop()

# ── Carga y vectorizacion (cacheada) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def build_chain(api_key: str):
    loader = TextLoader("DOSSIER ACTIVIDADES ROBOTICLAB.txt", encoding="utf-8")
    docs = loader.load()

    spl = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = spl.split_documents(docs)

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vs = FAISS.from_documents(chunks, emb)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(model="llama3-8b-8192", api_key=api_key, temperature=0.2)

    prompt = PromptTemplate.from_template("""
Eres un asistente util de RoboticLab. Responde en espanol usando unicamente
la informacion del contexto proporcionado. Si no encuentras la respuesta, dilo claramente.

Contexto:
{context}

Pregunta: {question}

Respuesta:""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chain = build_chain(GROQ_API_KEY)

# ── Historial de chat ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if pregunta := st.chat_input("Escribe tu pregunta sobre RoboticLab..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Pensando..."):
            try:
                respuesta = chain.invoke(pregunta)
            except Exception as e:
                respuesta = f"Error al procesar tu pregunta: {e}"
        st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 RoboticLab Assistant")
    st.markdown("Asistente basado en IA que responde preguntas sobre las actividades y servicios de RoboticLab.")
    st.divider()
    st.markdown("**Modelo:** LLaMA 3 (Groq)")
    st.markdown("**Embeddings:** Multilingual MPNet")
    st.markdown("**Busqueda:** FAISS vectorial")
    st.divider()
    if st.button("🗑️ Limpiar conversacion"):
        st.session_state.messages = []
        st.rerun()
