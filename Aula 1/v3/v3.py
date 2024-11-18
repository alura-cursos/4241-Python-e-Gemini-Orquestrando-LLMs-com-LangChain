from langchain_community.chat_models import ChatMaritalk
from langchain_core.messages import HumanMessage
from my_models import MARITACA_SABIA
from my_keys import MARITACA_API_KEY
from my_helper import encode_image

llm = ChatMaritalk(
    model=MARITACA_SABIA,
    api_key=MARITACA_API_KEY, 
    temperature=0.7,
    max_tokens=100,
)

resposta = llm.invoke("Qual a capital do Brasil?")
print(resposta.content)    

imagem = encode_image("Aula 1/v3/dados/exemplo_grafico.jpg")

pergunta = "Descreva o gráfico da imagem: "

mensagem = HumanMessage(
  content=[
    {'type': 'text', 'text': pergunta},
    {'type': 'image_url', 'image_url': f"data:image/jpeg;base64,{imagem}"}
  ]
)

resposta = llm.invoke([mensagem])