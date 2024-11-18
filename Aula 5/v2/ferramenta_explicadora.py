from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_PRO
from my_keys import GEMINI_API_KEY
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from detalhes_imagem_model import DetalhesImagemModelo
from my_helper import encode_image
import json
import ast
from langchain_community.chat_models import ChatMaritalk
from my_models import MARITACA_SABIA
from my_keys import MARITACA_API_KEY

class FerramentaExplicadoraAssuntos(BaseTool):
    name : str = "FerramentaExplicadoraAssuntos"
    description : str = """
    Utilize esta ferramenta sempre que for solicitado que você explique um conteúdo para jovens brasileiros.
    
    # Entradas Requeridas
    'tema' (str) : Tema do conteúdo a ser explicado.
    """
    return_direct : bool = True

    def _run(self, acao):
        acao = ast.literal_eval(acao)
        tema = acao.get("tema", "")
        
        llm = ChatMaritalk(
            model=MARITACA_SABIA,
            api_key=MARITACA_API_KEY, 
            temperature=0.7,
            max_tokens=4000,
        )

        template_resposta_final = PromptTemplate(
            template="""
            Elabore uma explicação sobre o tema {tema} que seja compreensível para jovens brasileiros.
            Utilize exemplos do cotidiano para ilustrar, e image que estão na idade do ensino médio.

            Se sugerir recursos, também leve em consideração o contexto brasileiro.

            Caso existam códigos, facilite na linguagem e use sempre python.

            Tema da pergunta: {tema}.
            """,
            input_variables=["resposta_analise_imagem"],
        )

        cadeia = template_resposta_final | llm | StrOutputParser()

        resposta_final = cadeia.invoke({"tema": tema})
        return resposta_final