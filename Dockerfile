
FROM python:3.11-slim

WORKDIR /api

# Copia primeiro o arquivo de dependências para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante dos arquivos da aplicação (do diretório 'src') para o contêiner
COPY src/ .

# Expõe a porta correta que a aplicação vai usar
EXPOSE 9592

# Comando para iniciar a aplicação
CMD ["gunicorn", "--bind", "0.0.0.0:9592", "api:api"]
