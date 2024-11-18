from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from my_models import GEMINI_PRO, GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY
from my_helper import encode_image


imagem = encode_image("Aula 1/v1/dados/exemplo_grafico.jpg")

llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_PRO)

pergunta = "Descreva o gráfico da imagem: "

mensagem = HumanMessage(
  content=[
    {'type': 'text', 'text': pergunta},
    {'type': 'image_url', 'image_url': f"data:image/jpeg;base64,{imagem}"}
  ]
)

resposta = llm.invoke([mensagem])

print(resposta.content)
