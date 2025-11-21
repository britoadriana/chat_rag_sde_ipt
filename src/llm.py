# Modelos de llm
import os
from dotenv import load_dotenv 
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings # embeddings - na versão final armazenar os modelos no servidor
from langchain_qdrant import QdrantVectorStore, RetrievalMode # Qdrant
from langchain_qdrant.fastembed_sparse import FastEmbedSparse # Qdrant
from fastembed import (
  SparseTextEmbedding,
  TextEmbedding,
)
import concurrent.futures

# from langchain_openai import ChatOpenAI 
# from langchain_groq import ChatGroq 

load_dotenv() #lembrar para poder ler o .env

# FORMAS SEPARADAS DE CHAMAR LLM: CHAMADA POR CADA GERENCIADOR

# Chaves usando Groq
# try:
#     groq_api_key = os.getenv("GROQ_API_KEY")
#     os.environ["GROQ_API_KEY"] = groq_api_key
#     print("Chave da API GROQ configurada com sucesso.")
# except (ImportError, KeyError) as e:
#     print("AVISO: Inserir chave de válida")
    
# llm = ChatGroq(
#     model="llama-3.1-8b-instant", # Modelo de geração mais leve, llama-3.1-8b-instant
#     temperature=0.1,
#     groq_api_key=os.getenv("GROQ_API_KEY"),
# )

# Chaves usando Openai
# try:
#     openai_api_key = os.getenv("OPENAI_API_KEY") 
#     os.environ["OPENAI_API_KEY"] = openai_api_key
#     print("Chave da API Openai configurada com sucesso.")
# except (ImportError, KeyError) as e:
#     print("AVISO: Inserir chave de válida")
    
# llm = ChatOpenAI(f
#     model="gpt-5-nano", 
#     temperature=0.1,
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
# )

# Mesmo docker modelo e código

# Definir pasta onde os modelos serão armazenados no HD
MODEL_CACHE_DIR = "./model_cache"
 
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE_DIR
os.environ["FASTEMBED_CACHE_PATH"] = os.path.join(MODEL_CACHE_DIR, "fastembed")

print("Cache configurado para:", MODEL_CACHE_DIR)
# Modelo denso
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("Modelo denso configurado com sucesso")
# Modelo esparso
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") #função matemática BM25, que classifica documentos com base na relevância em relação a uma consulta de pesquisa.
print("Modelo esparso configurado com sucesso")

# Mesmo docker modelo e código
# llm = ChatOllama(
#     model="llama3:8b", 
#     temperature=0.1,
#     base_url="http://localhost:11434"
# )

# print("LLM (Ollama) rodando via Docker (no host) configurado.")

# FORMA HÍBRIDA DE CHAMAR LLM: TENTATIVAS SEQUENCIAIS DE RODAR LLM POR CADA GERENCIADOR, 1 POR VEZ

def mostrar_mensagem_notebook():
    print("""
Oops! Estamos com problemas por aqui. Por favor, tente mais tarde.
 
Enquanto isso, você pode usar o nosso NotebookLM:
https://notebooklm.google.com/notebook/93d397f0-204b--4d55-93ee-8a609a6a1c79?authuser=3
 
Lá você pode explorar não só o conteúdo dos cadernos desenvolvidos pelo IPT e SDE,
mas também gerar mapas mentais, resumos em áudio e vídeo.
""")
 
 
def carregar_llm():
 
    # 1) TENTAR OLLAMA PRIMEIRO
    try:
        print("Tentando iniciar Ollama...")
 
        llm = ChatOllama(
            base_url="http://localhost:11434",  # ou localhost
            model="llama3:8b",
            temperature=0.1,
        )
 
        def testar_ollama():
            return llm.invoke("Explique em uma frase o que é inteligência artificial.")
       
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(testar_ollama)
            future.result(timeout=30)  # 30 segundos
 
        print("Ollama carregado com sucesso.")
        return llm
 
    except Exception as e_ollama:
        print("Ollama falhou:", str(e_ollama))
 
 
    # 2) SE OLLAMA FALHAR → VERIFICA CHAVE OPENAI
    openai_key = os.getenv("OPENAI_API_KEY")
 
    if not openai_key:
        print("Nenhuma chave OPENAI_API_KEY encontrada.")
        mostrar_mensagem_notebook()
        raise RuntimeError("Nenhum modelo disponível.")
 
 
    # 3) TENTAR OPENAI COMO SEGUNDA OPÇÃO
    try:
        print("Tentando fallback para OpenAI...")
 
        llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0.1,
            openai_api_key=openai_key,
        )
 
        print("OpenAI carregado com sucesso.")
        return llm
 
    except Exception as e_openai:
        print("OpenAI também falhou:", str(e_openai))
        mostrar_mensagem_notebook()
        raise RuntimeError("Nenhum modelo disponível.")
      
# llm = carregar_llm()
# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg)
