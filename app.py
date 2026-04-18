import streamlit as st
import os
import base64
from PIL import Image
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

BASE = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE, "logo.webp")
logo_img = Image.open(LOGO_PATH)
with open(LOGO_PATH, "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()
logo_src = f"data:image/webp;base64,{logo_b64}"

st.set_page_config(page_title="RoboticLab", page_icon=logo_img, layout="centered")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; font-family: 'Inter', sans-serif; }}

/* Reset Streamlit chrome */
#MainMenu, footer, header, .stDeployButton {{ display: none !important; }}
.block-container {{ padding: 0 !important; max-width: 760px !important; }}
.stApp {{ background: #0f0f0f; }}

/* Navbar */
.navbar {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 24px;
    border-bottom: 1px solid #1e1e1e;
    position: sticky;
    top: 0;
    background: #0f0f0f;
    z-index: 100;
}}
.navbar img {{ width: 32px; height: 32px; border-radius: 6px; object-fit: cover; }}
.navbar-title {{ font-size: 1rem; font-weight: 600; color: #f5f5f5; }}
.navbar-sub {{ font-size: 0.75rem; color: #555; margin-left: auto; }}

/* Chat container */
.chat-wrap {{ padding: 24px 24px 100px; }}

/* Empty state */
.empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px;
    gap: 12px;
}}
.empty-state img {{ width: 48px; height: 48px; border-radius: 10px; opacity: 0.6; }}
.empty-state h2 {{ font-size: 1.3rem; font-weight: 600; color: #ccc; margin: 0; }}
.empty-state p {{ font-size: 0.88rem; color: #555; margin: 0; text-align: center; }}

/* Mensajes */
.msg-row {{
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    align-items: flex-start;
}}
.msg-row.user {{ flex-direction: row-reverse; }}

.avatar {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    overflow: hidden;
}}
.avatar img {{ width: 100%; height: 100%; object-fit: cover; }}
.avatar.user-av {{
    background: #2a2a2a;
    color: #888;
    border: 1px solid #2e2e2e;
}}

.bubble {{
    max-width: 82%;
    padding: 11px 15px;
    border-radius: 14px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #e8e8e8;
}}
.bubble.bot {{
    background: #161616;
    border: 1px solid #202020;
    border-radius: 4px 14px 14px 14px;
}}
.bubble.user {{
    background: #1a2a3d;
    border: 1px solid #1e3450;
    border-radius: 14px 4px 14px 14px;
    color: #d8eaff;
}}

/* Input */
.stChatInput > div {{
    background: #161616 !important;
    border: 1px solid #282828 !important;
    border-radius: 12px !important;
    box-shadow: 0 0 0 0 transparent !important;
}}
.stChatInput > div:focus-within {{
    border-color: #333 !important;
    box-shadow: none !important;
}}
.stChatInput textarea {{
    background: transparent !important;
    color: #e8e8e8 !important;
    font-size: 0.9rem !important;
    caret-color: #e8e8e8 !important;
}}
.stChatInput textarea::placeholder {{ color: #444 !important; }}
section[data-testid="stBottom"] > div {{
    background: #0f0f0f !important;
    padding: 12px 24px 20px !important;
    border-top: 1px solid #1a1a1a;
}}

/* Ocultar avatares nativos de Streamlit */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {{ display: none !important; }}
[data-testid="stChatMessage"] {{
    background: transparent !important;
    padding: 0 !important;
}}
[data-testid="stChatMessageContent"] {{ background: transparent !important; padding: 0 !important; }}

/* Spinner */
.stSpinner p {{ color: #444 !important; font-size: 0.82rem !important; }}
</style>

<div class="navbar">
    <img src="{logo_src}" alt="logo">
    <span class="navbar-title">RoboticLab Assistant</span>
    <span class="navbar-sub">LLaMA 3.1 · Groq</span>
</div>
""", unsafe_allow_html=True)

# ── RAG chain ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def build_chain(api_key: str):
    txt = os.path.join(BASE, "DOSSIER ACTIVIDADES ROBOTICLAB.txt")
    loader = TextLoader(txt, encoding="utf-8")
    docs = loader.load()
    spl = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = spl.split_documents(docs)
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    vs = FAISS.from_documents(chunks, emb)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.2)
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

# ── Historial ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

col_spacer, col_btn = st.columns([6, 1])
with col_btn:
    if st.button("Limpiar", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.messages:
    st.markdown(f"""
    <div class="empty-state">
        <img src="{logo_src}" alt="logo">
        <h2>¿En qué puedo ayudarte?</h2>
        <p>Pregunta sobre actividades, talleres y servicios de RoboticLab</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user-av">TÚ</div>
            <div class="bubble user">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar"><img src="{logo_src}" alt="bot"></div>
            <div class="bubble bot">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
if pregunta := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})

    st.markdown(f"""
    <div class="msg-row user">
        <div class="avatar user-av">TÚ</div>
        <div class="bubble user">{pregunta}</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Pensando..."):
        try:
            respuesta = chain.invoke(pregunta)
        except Exception as e:
            respuesta = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": respuesta})

    st.markdown(f"""
    <div class="msg-row">
        <div class="avatar"><img src="{logo_src}" alt="bot"></div>
        <div class="bubble bot">{respuesta}</div>
    </div>""", unsafe_allow_html=True)
