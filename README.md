# chat_rag_sde_ipt
Chat sobre cadernos técnicos relacionados a cidades inteligentes desenvolvidos por IPT e SDE

## Ambiente
    - Python: 3.12.12
    - Ollama: 0.13.1-rc0
    - Docker: 28.1.1

## Como instalar as dependências em um ambiente python local
- **Comandos:** 
    - **Atualizar pip do ambiente:** python -m pip install -U pip
    - **Instalar via pip:** pip install -r requirements.txt

## Como criar o container e rodar em seguida
Rodar os seguintes comando em um terminal de uma máquina com docker instalado:
- **Criar container do modelo llama gerenciado em ollama:** 
    - Comando deve ser feito na pasta ollama
    - **Comando:** docker build -t ollama-model .
    - **Rodar container ollama após criado:** docker run -p 11434:11434 ollama-model
- **Criar container da aplicação em python:** 
    - Comando deve ser feito na pasta src
    - **Comando:** docker build -t assistente-cidades .
    - **Rodar container python após criado:** docker run -p 9593:9593 assistente-cidades
