#!/bin/sh
# Start ollama server in background
ollama serve & 
PID=$!

# Wait until server responds (timeout ~60s)
i=0
until ollama stats >/dev/null 2>&1 || [ $i -ge 60 ]; do
  i=$((i+1))
  sleep 1
done

# Pull model if not present
if ! ollama list | grep -q '^llama3:8b'; then
  ollama pull llama3:8b
fi

# Keep container running attached to the server process
wait $PID