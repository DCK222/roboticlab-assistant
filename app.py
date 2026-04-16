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

# ── Logo como base64 para HTML y PIL para avatares ────────────────────────────
LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo.webp")
logo_img = Image.open(LOGO_PATH)
with open(LOGO_PATH, "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()
logo_src = f"data:image/webp;base64,{logo_b64}"

st.set_page_config(
    page_title="RoboticLab Assistant",
    page_icon=logo_img,
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background: #1a1a1a; color: #ececec; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stChatInput textarea {
    background: #2d2d2d !important;
    border: 1px solid #3d3d3d !important;
    border-radius: 12px !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
}
.stChatInput textarea:focus {
    border-color: #555 !important;
    box-shadow: none !important;
}
.stChatInput > div {
    background: #2d2d2d !important;
    border: 1px solid #3d3d3d !important;
    border-radius: 12px !important;
}

.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
}

[data-testid="stChatMessageContent"] p {
    font-size: 0.95rem;
    line-height: 1.65;
    color: #ececec;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #2d2d2d !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid #3d3d3d !important;
    color: #888 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 4px 14px !important;
    transition: all 0.2s;
}
.stButton > button:hover {
    border-color: #666 !important;
    color: #ccc !important;
}

.stTextInput input {
    background: #2d2d2d !important;
    border: 1px solid #3d3d3d !important;
    border-radius: 10px !important;
    color: #ececec !important;
}
.stTextInput input:focus {
    border-color: #666 !important;
    box-shadow: none !important;
}

hr { border-color: #2d2d2d !important; }
.stSpinner > div { border-top-color: #666 !important; }
</style>
""", unsafe_allow_html=True)

# ── API Key ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = (
    st.secrets.get("GROQ_API_KEY", "")
    or os.getenv("GROQ_API_KEY", "")
)

# ── RAG chain (cacheada) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def build_chain(api_key: str):
    txt = os.path.join(os.path.dirname(__file__), "DOSSIER ACTIVIDADES ROBOTICLAB.txt")
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

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title, col_btn = st.columns([0.6, 5, 1.4])
with col_logo:
    st.image(logo_img, width=44)
with col_title:
    st.markdown("<h3 style='margin:0; padding-top:4px; color:#ececec'>RoboticLab Assistant</h3>", unsafe_allow_html=True)
with col_btn:
    if st.button("Nueva chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown("<hr style='margin: 0.5rem 0 1rem'>", unsafe_allow_html=True)

# ── Historial ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(f"""
    <div style='text-align:center; padding:4rem 2rem; color:#555'>
        <img src='{logo_src}' style='width:56px; opacity:0.4; margin-bottom:1rem'>
        <div style='font-size:1.1rem; font-weight:500; color:#666'>¿En qué puedo ayudarte?</div>
        <div style='font-size:0.85rem; margin-top:0.4rem'>Pregunta sobre actividades, talleres o servicios de RoboticLab</div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = logo_img if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if pregunta := st.chat_input("Pregunta algo sobre RoboticLab..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar=logo_img):
        with st.spinner(""):
            try:
                respuesta = chain.invoke(pregunta)
            except Exception as e:
                respuesta = f"Error: {e}"
        st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})
