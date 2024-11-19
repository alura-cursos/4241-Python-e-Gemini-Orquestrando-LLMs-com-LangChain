from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from langchain.globals import set_debug
set_debug(False)

from langchain import hub
from langchain.agents import create_react_agent
from langchain.agents import Tool
from ferramenta_analisadora_imagem import FerramentaAnalisadoraImagem
from ferramenta_explicadora import FerramentaExplicadora

class AgenteOrquestrador:
  def __init__(self):
    self.llm = ChatGoogleGenerativeAI(
      api_key=GEMINI_API_KEY,
      model=GEMINI_FLASH
    )

    ferramenta_analisadora_imagem = FerramentaAnalisadoraImagem()
    ferrameta_explicadora = FerramentaExplicadora()

    self.tools = [
      Tool(
        name = ferramenta_analisadora_imagem.name,
        func = ferramenta_analisadora_imagem.run,
        description = ferramenta_analisadora_imagem.description,
        return_direct = ferramenta_analisadora_imagem.return_direct
      ),
      Tool(
        name = ferrameta_explicadora.name,
        func = ferrameta_explicadora.run,
        description = ferrameta_explicadora.description,
        return_direct = ferrameta_explicadora.return_direct
      )
    ]

    prompt = hub.pull("hwchase17/react")
    self.agente = create_react_agent(self.llm, self.tools, prompt)