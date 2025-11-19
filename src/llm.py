# Modelos de llm
import os
from dotenv import load_dotenv 
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings # embeddings - na versão final armazenar os modelos no servidor
from langchain_qdrant import QdrantVectorStore, RetrievalMode # Qdrant
from langchain_qdrant.fastembed_sparse import FastEmbedSparse # Qdrant
from fastembed import (
  SparseTextEmbedding,
  TextEmbedding,
)

load_dotenv() #lembrar para poder ler o .env

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
llm = ChatOllama(
    model="llama3:8b", 
    temperature=0.1,
    # Ele vai procurar por "http://localhost:11434" por padrão,
    # e o Docker vai redirecionar para o container.
)
print("LLM (Ollama) rodando via Docker (no host) configurado.")

# if __name__ == '__main__':
#     # Bloco de teste para ser executado apenas quando o script é chamado diretamente
#     print("Executando teste de invocação do LLM...")
#     messages = [
#         (
#             "system",
#             "You are a helpful assistant that translates English to French. Translate the user sentence.",
#         ),
#         ("human", "I love programming."),
#     ]
#     ai_msg = llm.invoke(messages)
#     print(ai_msg)