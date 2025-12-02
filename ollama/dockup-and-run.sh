docker container stop PCISDE.OLLAMA
docker container rm PCISDE.OLLAMA
docker image rm ollama-model
docker build -t ollama-model .
nohup docker run --name PCISDE.OLLAMA -p 11434:11434 ollama-model &