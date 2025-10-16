import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ==============================================
# CONFIGURAÇÕES INICIAIS
# ==============================================

print("🚀 Iniciando processamento dos POPs...")

# Carrega variáveis do .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ ERRO: OPENAI_API_KEY não encontrada. Verifique seu arquivo .env.")
    exit(1)
else:
    print("🔑 OPENAI_API_KEY carregada com sucesso!")

# Pastas principais
DATA_DIR = "data"
OUT_DIR = "base_pop"

# Cria pasta de saída se não existir
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================
# FUNÇÕES AUXILIARES
# ==============================================

def carregar_docs(pasta=DATA_DIR):
    """Carrega todos os PDFs da pasta data/"""
    print(f"📂 Carregando documentos da pasta '{pasta}'...")
    loader = DirectoryLoader(pasta, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"✅ {len(docs)} documentos/páginas pré-carregados. Fazendo chunking...")
    return docs


def dividir_textos(docs):
    """Divide os textos em chunks menores para o embedding"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    textos = splitter.split_documents(docs)
    print(f"🧩 {len(textos)} chunks gerados.")
    return textos


def gerar_base_vectorial():
    """Gera a base vetorial FAISS a partir dos documentos PDF"""
    try:
        docs = carregar_docs()
        textos = dividir_textos(docs)

        print("⚙️ Gerando embeddings (OpenAI) e criando base FAISS...")

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )

        base = FAISS.from_documents(textos, embeddings)

        print("💾 Salvando base vetorial...")
        base.save_local(OUT_DIR)

        print("✅ Base vetorial criada e salva em:", OUT_DIR)

    except Exception as e:
        print("❌ Erro ao gerar base vetorial:", str(e))


# ==============================================
# EXECUÇÃO PRINCIPAL
# ==============================================

if __name__ == "__main__":
    gerar_base_vectorial()
