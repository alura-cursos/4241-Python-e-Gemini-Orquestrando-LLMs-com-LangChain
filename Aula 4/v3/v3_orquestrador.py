from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent
from langchain import hub
from langchain.agents import Tool
import os
from langchain.globals import set_debug
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY

from v3_ferramenta_analise_imagem import FerramentaAnaliseImagem

class AgenteOrquestrador:
    def __init__(self):
        llm_gemini = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_FLASH)

        ferramenta_analisadora_imagem = FerramentaAnaliseImagem()
        
        self.tools = [
            Tool(
                name = ferramenta_analisadora_imagem.name,
                func = ferramenta_analisadora_imagem.run,
                description = ferramenta_analisadora_imagem.description,
                return_direct = ferramenta_analisadora_imagem.return_direct
                ),
        ]

        prompt = hub.pull("hwchase17/react")
        self.agente = create_react_agent(llm_gemini, self.tools, prompt)
