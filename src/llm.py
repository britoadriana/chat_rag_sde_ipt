# # Para que serve este arquivo:
# # Aqui fica a lógica de carregar e interagir com o seu Modelo de Linguagem (LLM).
# # A função `get_llm_response` é um exemplo que simula a resposta do modelo.
# # Você deve substituir essa simulação pela lógica real de inferência do seu LLM.

# # Como modificar conforme seu projeto:
# # 1. Escolha seu framework (LangChain, Transformers, etc.) e importe as bibliotecas.
# # 2. Carregue seu modelo (seja de um arquivo local, do Hugging Face Hub, ou de uma API).
# #    É uma boa prática carregar o modelo apenas uma vez, quando a aplicação inicia,
# #    para evitar sobrecarga.
# # 3. Implemente a função `get_llm_response` para que ela receba o prompt, passe-o
# #    para o modelo e retorne o texto gerado.

# async def get_llm_response(prompt: str) -> str:
#     """
#     Processa o prompt e retorna a resposta do LLM.

#     Esta é uma implementação de exemplo. Substitua pelo seu código de inferência.
#     """
#     # TODO: Substitua esta lógica de simulação pela chamada real ao seu LLM.
#     # Exemplo com LangChain (descomente e adapte):
#     # from langchain_community.llms import Ollama
#     # llm = Ollama(model="llama2")
#     # return llm.invoke(prompt)

#     # Exemplo com Transformers (descomente e adapte):
#     # from transformers import pipeline
#     # generator = pipeline('text-generation', model='gpt2')
#     # result = generator(prompt, max_length=50)
#     # return result[0]['generated_text']

#     print(f"Recebido prompt: {prompt}")
#     response = f"Esta é uma resposta simulada para o prompt: '{prompt}'"
#     print(f"Gerada resposta: {response}")
#     return response

# Modelos de llm
import os
from dotenv import load_dotenv 
from langchain_huggingface import HuggingFaceEmbeddings # embeddings - na versão final armazenar os modelos no servidor
from langchain_groq import ChatGroq # llm - versão final usar olhama para colocar o modelo
from langchain_qdrant import QdrantVectorStore, RetrievalMode # Qdrant
from langchain_qdrant.fastembed_sparse import FastEmbedSparse # Qdrant
from fastembed import (
  SparseTextEmbedding,
  TextEmbedding,
)

load_dotenv() #lembrar para poder ler o .env

# Configurar Chaves de API do modelo
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = groq_api_key
    print("Chave da API GROQ configurada com sucesso.")
except (ImportError, KeyError) as e:
    print("AVISO: Inserir chave de válida")
    
llm = ChatGroq(
    model="llama-3.1-8b-instant", # Modelo de geração mais leve, llama-3.1-8b-instant
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# Modelo denso
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("Modelo denso configurado com sucesso")
# Modelo esparso
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") #função matemática BM25, que classifica documentos com base na relevância em relação a uma consulta de pesquisa.
print("Modelo esparso configurado com sucesso")
