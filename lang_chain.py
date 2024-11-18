from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
  api_key=GEMINI_API_KEY,
  model=GEMINI_FLASH
)

imagem = encode_image("dados\exemplo_grafico.jpg")

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

cadeia = template_analisador | llm | StrOutputParser()
resposta = cadeia.invoke({"imagem_informada": imagem})

print(resposta)