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
# from langchain_openai import ChatOpenAI # acrescentar em dependências
# from langchain_groq import ChatGroq # llm 


load_dotenv() #lembrar para poder ler o .env

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
llm = ChatOllama(
    model="llama3:8b", 
    temperature=0.1,
    # Ele vai procurar por "http://localhost:11434" por padrão,
    # e o Docker vai redirecionar para o container.
)
print("LLM (Ollama) rodando via Docker (no host) configurado.")

# Modelo denso
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
# print("Modelo denso configurado com sucesso")
# # Modelo esparso
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") #função matemática BM25, que classifica documentos com base na relevância em relação a uma consulta de pesquisa.
print("Modelo esparso configurado com sucesso")


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