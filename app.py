import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# ==============================================
# CONFIGURAÇÕES INICIAIS
# ==============================================

st.set_page_config(
    page_title="🔌 Neoenergia POP Finder",
    page_icon="⚡",
    layout="centered",
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY não encontrada! Verifique seu arquivo `.env`.")
    st.stop()

BASE_DIR = "base_pop"

# ==============================================
# ESTILO VISUAL (Mobile e Branding Neoenergia)
# ==============================================

st.markdown("""
<style>
/* Estilo principal */
body, .stApp {
    background-color: #f9fdf9;
    color: #003A1B;
    font-family: 'Segoe UI', sans-serif;
}

/* Caixa da pergunta */
div[data-baseweb="input"] > div {
    border: 2px solid #009739 !important;
    border-radius: 12px !important;
}

/* Botão */
button[kind="primary"] {
    background-color: #009739 !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
}

/* Título */
h1, h2, h3 {
    color: #006341;
}

/* Ajuste mobile */
@media (max-width: 768px) {
    .stApp {
        padding: 10px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ==============================================
# CARREGAR BASE VETORIAL
# ==============================================

@st.cache_resource(show_spinner="📂 Carregando base de conhecimento...")
def carregar_base():
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        base = FAISS.load_local(BASE_DIR, embeddings, allow_dangerous_deserialization=True)
        return base
    except Exception as e:
        st.error(f"❌ Erro ao carregar base vetorial: {str(e)}")
        return None

base = carregar_base()
if not base:
    st.stop()

retriever = base.as_retriever(search_kwargs={"k": 5})

# ==============================================
# CONFIGURAÇÃO DO MODELO
# ==============================================

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

# ==============================================
# PROMPT OTIMIZADO (versão operacional)
# ==============================================

prompt_template = """
Você é o **Assistente Técnico Operacional Neoenergia**, especialista em segurança e execução de tarefas de campo.
Sua função é **responder perguntas operacionais com base nos POPs oficiais** da empresa, de forma **direta, segura e confiável**.

Siga as diretrizes:
- Use linguagem **simples, técnica e objetiva**, adequada a técnicos e eletricistas de campo.
- Estruture a resposta em **passos numerados ou tópicos curtos**.
- Destaque **EPIs, ferramentas, riscos e procedimentos obrigatórios**.
- Se aplicável, cite o **nome do POP** e código correspondente.
- Caso a informação não esteja disponível, diga: “Não há orientação específica sobre isso nos POPs disponíveis.”
- Nunca invente informações fora dos documentos.

Contexto técnico (trechos dos POPs):
{context}

Pergunta do colaborador:
{question}

Responda de forma clara e segura:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=modelo,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ==============================================
# INTERFACE STREAMLIT
# ==============================================

st.image("https://upload.wikimedia.org/wikipedia/commons/a/a2/Neoenergia_logo.png", width=160)
st.title("⚡ Neoenergia POP Finder")
st.markdown("""
### Assistente Técnico Operacional  
Consulte rapidamente procedimentos, requisitos e diretrizes dos POPs diretamente do seu celular.
""")

st.divider()

pergunta = st.text_input(
    "💬 Digite sua dúvida técnica:",
    placeholder="Ex: Qual o procedimento correto para escalada em poste metálico?"
)

col1, col2 = st.columns([1, 1])
with col1:
    buscar = st.button("🔍 Consultar POPs")
with col2:
    limpar = st.button("🧹 Limpar")

if limpar:
    st.experimental_rerun()

if buscar:
    if not pergunta.strip():
        st.warning("Por favor, digite uma pergunta primeiro.")
    else:
        with st.spinner("🔎 Buscando resposta nos POPs..."):
            resposta = qa_chain.invoke({"query": pergunta})

            st.success("✅ Resposta Técnica:")
            st.markdown(resposta["result"])

            fontes = resposta.get("source_documents", [])
            if fontes:
                with st.expander("📄 Fontes consultadas"):
                    for doc in fontes:
                        origem = os.path.basename(doc.metadata.get("source", "Desconhecido"))
                        st.markdown(f"- **{origem}**")

st.markdown("---")
st.caption("🔋 Neoenergia POP Finder • IA Corporativa de Suporte Operacional (versão mobile PRO)")
