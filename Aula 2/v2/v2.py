from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from my_models import GEMINI_PRO, GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

imagem = encode_image("Aula 2/v2/dados/exemplo_grafico.jpg")

llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_PRO)

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

cadeia = template_analise_imagens | llm | parseador_saida_string

resposta = cadeia.invoke({"dados_imagem" : imagem})

print(resposta)
