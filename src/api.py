
import os
import uuid

from flask import Flask, request, jsonify
from agent import generate_response_with_guardrails

# Cria a instância da aplicação Flask
api = Flask(__name__)

@api.route('/processar', methods=['POST'])
def processar_endpoint():

    # Valida se o request contém um JSON
    if not request.is_json:
        return jsonify({"error": "Requisição inválida. O corpo deve ser um JSON."}), 400

    # Extrai os dados do JSON
    data = request.get_json()
    input_query = data.get('input_query', None)

    # Valida se a chave 'input_query' existe e não está vazia
    if not input_query:
        return jsonify({"error": "A chave 'input_query' está faltando ou está vazia no JSON."}), 400

    try:
        # Chama a rotina de processamento com a query recebida
        session_id = str(uuid.uuid4())
        response_string = generate_response_with_guardrails(input_query, session_id)

        # Prepara e retornar a resposta em formato JSON
        return jsonify({"resposta": response_string})

    except Exception as e:
        # Captura de erro genérica para problemas durante o processamento
        api.logger.error(f"Erro ao processar a query: {e}")
        return jsonify({"error": "Ocorreu um erro interno no servidor."}), 500

# Permite executar o servidor diretamente com 'python api.py'
if __name__ == '__main__':
    api.run(host='0.0.0.0', debug=True, port=9593)
