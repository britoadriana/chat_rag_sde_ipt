FROM python:3.11-slim

WORKDIR /api

# Copia primeiro o arquivo de dependências para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Instala ferramentas necessárias para executar o instalador do ollama em runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copia o restante dos arquivos da aplicação (do diretório 'src') para o contêiner
COPY src/ .

# Expõe a porta correta que a aplicação vai usar
EXPOSE 9593

# Ao iniciar: executa o instalador do ollama, puxa o modelo e então inicia o gunicorn
CMD ["/bin/sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh && ollama pull llama3:8b && exec gunicorn --bind 0.0.0.0:9593 api:api"]
