import os
from dotenv import load_dotenv

class ConfigDB:
    def __init__(self):
        # Carrega as variáveis do arquivo .env para o os.environ
        variaveis_env = load_dotenv()
        if not variaveis_env:
            print("Aviso: Arquivo .env não encontrado ou vazio.")

    def obter_configs(self, banco_dados):

        try:
            env = os.getenv("ENV")

            # Padroniza para maiúsculas para coincidir com o .env
            bd_prefix_url =  f"{banco_dados.upper()}_URL_{env.upper()}"
            bd_prefix_api_key =  f"{banco_dados.upper()}_API_KEY_{env.upper()}"
            
            # Busca dinamicamente as variáveis de ambiente
            url = os.getenv(f"{bd_prefix_url}")
            api_key = os.getenv(f"{bd_prefix_api_key}")
            
            return env, url, api_key
        
        except Exception as e:
            return None, None, None