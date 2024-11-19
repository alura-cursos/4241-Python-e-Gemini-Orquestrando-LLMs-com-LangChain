from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatMaritalk
from my_models import MARITACA_SABIA
from my_keys import MARITACA_API_KEY
import ast

class FerramentaExplicadora(BaseTool):
  name : str = "FerramentaExplicadora"
  description : str = """
  Utilize esta ferramenta sempre que for solicitadoq ue você explique um conteúdo
  para pessoas.

  # Entrada Requeridas
  - 'tema' (str) : Tema principal informado na pergunta do usuário
  """

  return_direct : bool = True

  def _run(self, acao):
    acao = ast.literal_eval(acao)
    tema_parametro = acao.get("tema", "")

    llm = ChatMaritalk(
      api_key=MARITACA_API_KEY,
      model=MARITACA_SABIA
    )

    template_resposta = PromptTemplate(
      template="""
      Assuma o papel de um professor preocupado com aspectos de didática do usuário.

      1. Elabore uma explicação sobre o tema {tema} que seja compreensível por estudantes na fase de conclusão do ensino médio.
      2. Utilize exemplos do cotidiano para tornar a explicação mais fácil
      3. Caso sugira algum recurso para apoiar a explicação, lembre-se do cenário e contexto brasileiro.
      4. Caso você apresente um código, seja didático e utilize Python

      Tema pergunta: {tema}
      """,
      input_variables=["tema"]
    )

    cadeia = template_resposta | llm | StrOutputParser()
    resposta = cadeia.invoke({"tema" : tema_parametro})
    return resposta