from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from detalhes_imagem_modelo import DetalhesImagemModelo
import ast


class FerramentaAnalisadoraImagem(BaseTool):
  name:str = "FerramentaAnalisadoraImagem"
  description:str = """
  Utilize esta ferramenta sempre que for solicitado que você faça uma análise
  de imagem.

  # Entradas Requiridas
  - 'nome_imagem' (str) : Nome da imagem a ser analisada com extensão de JPG. Exemplo: teste.jpg ou teste.jpeg
  """
  return_direct : bool = False

  def _run(self, acao):
    acao = ast.literal_eval(acao)
    caminho_imagem = acao.get("nome_imagem", "")

    llm = ChatGoogleGenerativeAI(
      api_key=GEMINI_API_KEY,
      model=GEMINI_FLASH
    )

    imagem = encode_image(f"dados\{caminho_imagem}")

    template_analisador = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          """
          Assuma que você é um analisador de imagens. A sua tarefa principal
          consiste em: analisar uma imagem e extrair informações importantes
          de forma objetiva.

          # FORMATO DE SAÍDA
          Descrição da Imagem: 'Coloque a sua descrição da imagem aqui'
          Rótulos: 'Coloque uma lista com três termos chave separados por vírgula'
          """
        ),
        (
          "user",
          [
            {
              "type" : "text", 
              "text" : "Descreva a imagem: "
            },
            {
              "type" : "image_url",
              "image_url" : {"url":"data:image/jpeg;base64,{imagem_informada}"}
            }
          ]
        )
      ]
    )

    cadeia_analise_imagem = template_analisador | llm | StrOutputParser()

    parser_json_imagem = JsonOutputParser(
      pydantic_object=DetalhesImagemModelo
    )

    template_resposta = PromptTemplate(
      template="""
      Gere um resumo, utilizando uma linguagem clara e objetiva, focada
      no público brasileiro. A ideia é que a comunicação do resultado
      seja o mais fácil possível, priorizando registros para consultas
      posteriores.

      # O Resultado da imagem
      {resposta_cadeia_analise_imagem}

      # FORMATO DE SAÍDA
      {formato_saida}
      """,
      input_variables=["resposta_cadeia_analise_imagem"],
      partial_variables={
        "formato_saida" : parser_json_imagem.get_format_instructions()
      }
    )

    cadeia_resumo = template_resposta | llm | parser_json_imagem

    cadeia_completa = (cadeia_analise_imagem | cadeia_resumo)

    resposta = cadeia_completa.invoke({"imagem_informada": imagem})
    return resposta