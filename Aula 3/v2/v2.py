from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from my_models import GEMINI_PRO, GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatMaritalk
from langchain.globals import set_debug
set_debug(True)

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from detalhes_imagem_model import DetalhesImagemModelo

imagem = encode_image("Aula 3/v2/dados/exemplo_grafico.jpg")

llm_gemini = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_PRO)

template_analise_imagens = ChatPromptTemplate.from_messages(
    [
        (
          "system",  
          """Assuma que você é um analisador de imagens experiente, e que sua tarefa é criar rótulos 
          inteligentes que ajudem a descrever e a encontrar imagens salvas.

          # FORMATO DE SAÍDA
          Descrição da Imagem: 'coloque aqui  descrição da imagem'
          Rótulos: 'Coloque aqui uma lista com três tags que representem essa imagem e tornem fácil sua identificação.'
          """
        ),
        (
            "user",
            [
                {
                  "type": "text", 
                  "text": "Descreva a imagem abaixo e forneça detalhes relevantes para que possa ser registrada."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{dados_imagem}"},
                }
            ],
        ),
    ]
)

parseador_saida_string = StrOutputParser()

cadeia_imagem = template_analise_imagens | llm_gemini | parseador_saida_string

resposta_analise_imagem = cadeia_imagem.invoke({"dados_imagem" : imagem})

parseador_detalhes_estruturados = JsonOutputParser(pydantic_object=DetalhesImagemModelo)

resposta_final = PromptTemplate(
    template="""
    Gere um resumo para a imagem informada. Neste resumo leve em consideração o conteúdo da cadeia anterior: 
    {resposta_analise_imagem}.

    # FORMATO DE SAÍDA
    {formato_saida}
    """,
    input_variables=["resposta_analise_imagem"],
    partial_variables={"formato_saida": parseador_detalhes_estruturados.get_format_instructions()},
)

cadeia_resposta_final = resposta_final | llm_gemini | parseador_saida_string

cadeia_simples = (cadeia_imagem | cadeia_resposta_final)

resposta_final = cadeia_simples.invoke({"dados_imagem": imagem})