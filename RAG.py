import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Carga del TXT
print("Cargando documento...")
loader = TextLoader("DOSSIER ACTIVIDADES ROBOTICLAB.txt", encoding="utf-8")
docs = loader.load()
print(f"   {len(docs)} documento(s) cargados.")

# Fragmentacion
spl = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = spl.split_documents(docs)
print(f"   {len(chunks)} fragmentos generados.")

# Embeddings + Vector Store
print("Generando embeddings...")
emb = OllamaEmbeddings(model="nomic-embed-text")
vs = FAISS.from_documents(chunks, emb)
retriever = vs.as_retriever(search_kwargs={"k": 4})

# LLM
llm = OllamaLLM(model="llama3")

# Prompt
prompt = PromptTemplate.from_template("""
Eres un asistente util. Responde en espanol usando unicamente
la informacion del contexto proporcionado. Si no encuentras la respuesta,
dilo claramente.

Contexto:
{context}

Pregunta: {question}

Respuesta:""")

# Chain
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Loop de conversacion
print("\nSistema listo. Escribe 'q' para salir.\n")

while True:
    pregunta = input("Tu: ").strip()
    if pregunta.lower() in ("q", "quit", "salir", "exit"):
        break
    if not pregunta:
        continue
    try:
        respuesta = chain.invoke(pregunta)
        print(f"\nBot: {respuesta}\n")
    except Exception as e:
        print(f"\nError: {e}\n")

print("Sesion finalizada.")
