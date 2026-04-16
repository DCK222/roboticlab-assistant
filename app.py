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

st.set_page_config(
    page_title="RoboticLab Assistant",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-logo {
    font-size: 4rem;
    line-height: 1;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00c6ff, #0072ff, #00c6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.3rem 0 0.2rem;
}
.hero-subtitle {
    color: #7a9cc0;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}

/* Chat messages */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.6rem 0;
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 0.6rem 0;
}
.bubble-user {
    background: linear-gradient(135deg, #0072ff, #00c6ff);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 70%;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(0,114,255,0.3);
}
.bubble-bot {
    background: linear-gradient(135deg, #1a2a3a, #1e3448);
    color: #e0eaf5;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    max-width: 70%;
    font-size: 0.95rem;
    border: 1px solid #2a4060;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    margin: 0 8px;
    flex-shrink: 0;
    align-self: flex-end;
}
.avatar-bot { background: linear-gradient(135deg, #0072ff, #00c6ff); }
.avatar-user { background: linear-gradient(135deg, #444, #666); }

/* Input */
.stChatInput > div {
    background: #1a2a3a !important;
    border: 1px solid #2a4060 !important;
    border-radius: 12px !important;
}

/* Sidebar */
.sidebar-card {
    background: linear-gradient(135deg, #1a2a3a, #1e3448);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #2a4060;
    margin-bottom: 1rem;
}
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #0072ff22, #00c6ff22);
    border: 1px solid #0072ff55;
    color: #00c6ff;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-logo">🤖</div>
    <div class="hero-title">RoboticLab Assistant</div>
    <div class="hero-subtitle">Asistente inteligente · Powered by LLaMA 3 + Groq</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
sidebar, chat_col = st.columns([1, 3])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <strong style="color:#00c6ff">⚡ Tecnologia</strong><br><br>
        <span class="badge">LLaMA 3 8B</span>
        <span class="badge">Groq Cloud</span>
        <span class="badge">FAISS</span>
        <span class="badge">LangChain</span>
        <span class="badge">HuggingFace</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <strong style="color:#00c6ff">📚 Sobre este asistente</strong><br><br>
        <span style="color:#7a9cc0; font-size:0.88rem">
        Responde preguntas sobre las actividades, talleres y servicios de RoboticLab
        usando busqueda vectorial sobre el dossier oficial.
        </span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Nueva conversacion", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── API Key ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

if not GROQ_API_KEY:
    with chat_col:
        st.warning("⚠️ Necesitas una GROQ API Key para usar el asistente.")
        GROQ_API_KEY = st.text_input("Ingresa tu GROQ API Key", type="password", placeholder="gsk_...")
        st.caption("Obtén una gratis en [console.groq.com](https://console.groq.com)")
    if not GROQ_API_KEY:
        st.stop()

# ── RAG chain (cacheada) ──────────────────────────────────────────────────────
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

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

chain = build_chain(GROQ_API_KEY)

# ── Chat ──────────────────────────────────────────────────────────────────────
with chat_col:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_area = st.container(height=480)

    with chat_area:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding:3rem 1rem; color:#3a5a7a;">
                <div style="font-size:3rem">💬</div>
                <div style="font-size:1rem; margin-top:0.5rem">
                    Pregunta algo sobre RoboticLab para comenzar
                </div>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="bubble-user">{msg["content"]}</div>
                    <div class="avatar avatar-user">🧑</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-bot">
                    <div class="avatar avatar-bot">🤖</div>
                    <div class="bubble-bot">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)

    if pregunta := st.chat_input("Escribe tu pregunta sobre RoboticLab..."):
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.spinner("Pensando..."):
            try:
                respuesta = chain.invoke(pregunta)
            except Exception as e:
                respuesta = f"Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        st.rerun()
