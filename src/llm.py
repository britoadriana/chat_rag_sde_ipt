# Modelos de llm
import os
from dotenv import load_dotenv 
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings # embeddings - na versão final armazenar os modelos no servidor
from langchain_qdrant import QdrantVectorStore, RetrievalMode # Qdrant
from langchain_qdrant.fastembed_sparse import FastEmbedSparse # Qdrant
from fastembed import (
  SparseTextEmbedding,
  TextEmbedding,
)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import concurrent.futures

# from langchain_openai import ChatOpenAI 
# from langchain_groq import ChatGroq 

load_dotenv() #lembrar para poder ler o .env

# Mesmo docker modelo e código

# Definir pasta onde os modelos serão armazenados no HD
MODEL_CACHE_DIR = "./model_cache"
 
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE_DIR
os.environ["FASTEMBED_CACHE_PATH"] = os.path.join(MODEL_CACHE_DIR, "fastembed")

print("Cache configurado para:", MODEL_CACHE_DIR)

# FORMA HÍBRIDA DE CHAMAR LLM: TENTATIVAS SEQUENCIAIS DE RODAR LLM POR CADA GERENCIADOR, 1 POR VEZ
 
def carregar_llm():
 
    # 1) TENTAR OLLAMA PRIMEIRO
    try:
        print("Tentando iniciar Ollama...")
 
        llm = ChatOllama(
            base_url="http://localhost:11434",  # ou localhost
            model="gemma3:4b",
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
        
 
    # # 2) SE OLLAMA FALHAR → VERIFICA CHAVE OPENAI
    # print("Tentando fallback para OpenAI...")
    # openai_key = os.getenv("OPENAI_API_KEY")

    # if not openai_key:
    #     print("Nenhuma chave OPENAI_API_KEY encontrada.")
    
    # else:
    #     try:

    #         llm = ChatOpenAI(
    #             model="gpt-5-nano",
    #             temperature=0.1,
    #             openai_api_key=openai_key,
    #         )

    #         print("OpenAI carregado com sucesso.")
    #         return llm

    #     except Exception as e_openai:
    #         print("OpenAI também falhou:", str(e_openai))
    
    # # 3) TENTAR GROQ COMO TERCEIRA OPÇÃO
    # print("Tentando fallback para Groq...")
    # groq_key = os.getenv("GROQ_API_KEY") 
    
    # if not groq_key:
    #     print("Nenhuma chave GROQ_API_KEY encontrada.")   
        
    # else:
    #     try:

    #         llm = ChatGroq(
    #             model="llama-3.1-8b-instant", # Modelo de geração mais leve, llama-3.1-8b-instant
    #             temperature=0.1,
    #             groq_api_key=os.getenv("GROQ_API_KEY"),
    #         )
            
    #         print("Groq carregado com sucesso.")
    #         return llm
            
    #     except Exception as e_groq:
    #         print("Groq também falhou:", str(e_groq))

    raise RuntimeError("Nenhum modelo disponível.")

# Modelo denso
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("Modelo denso configurado com sucesso")

# Modelo esparso
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") #função matemática BM25, que classifica documentos com base na relevância em relação a uma consulta de pesquisa.
print("Modelo esparso configurado com sucesso")

# Modelo Guardrails - Prompt Injection 
tokenizer = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2",
    cache_dir=MODEL_CACHE_DIR)

pi_model = AutoModelForSequenceClassification.from_pretrained(
    "protectai/deberta-v3-base-prompt-injection-v2",
    use_safetensors=True
)

# DECIDIR CPU OU GPU 
# device = "cuda" if torch.cuda.is_available() else "cpu"  # Escolhe automaticamente
device = "cpu"  # Forçar CPU
# device = "cuda"  # Forçar GPU

# Mover modelo para o dispositivo escolhido
model = pi_model.to(device)
print(f"Modelo guardrails carregado em: {device}")

# Teste da classe:
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
