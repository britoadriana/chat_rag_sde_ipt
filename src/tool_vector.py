import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Any
from llm import embedding_model, sparse_embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode # Qdrant
from qdrant_client import QdrantClient, models# hybrid dense/sparse store Qdrant
from langchain_core.prompts import ChatPromptTemplate # chains
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Carrega variáveis de ambiente
load_dotenv()

client = QdrantClient(
    url="http://10.11.39.33:6333"
)
print("Qdrant configurado com sucesso")

# Coleção onde estão os chunks no banco vetorial, mudar para local
collection_name = "SDE_REC"
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embedding_model,          # modelo denso
    sparse_embedding=sparse_embeddings, # modelo esparço
    retrieval_mode=RetrievalMode.HYBRID,#usar hibrido
    vector_name="dense",
    sparse_vector_name="sparse",
)

retriever = vector_store.as_retriever(
    search_type="similarity"
)

def find_chunk(query: str):
    """
    Busca os chunks de texto mais relevantes no índice vetorial.
    """
    rrf  = models.FusionQuery(fusion=models.Fusion.RRF) # modo de fusão de resultados pela busca hibrida
    docs_scores = vector_store.similarity_search_with_score( 
        query, k=4, hybrid_fusion=rrf
    )
    # context = "\n\n---\n\n".join([doc.page_content for doc in docs_scores])
    return docs_scores
 
# Teste
# query = "o que é cidade inteligente?"
# chunks_retornados = find_chunk(query)
# print(chunks_retornados)
    
