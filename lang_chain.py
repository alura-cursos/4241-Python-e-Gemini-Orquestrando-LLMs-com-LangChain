from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY
from my_helper import encode_image

llm = ChatGoogleGenerativeAI(
  api_key=GEMINI_API_KEY,
  model=GEMINI_FLASH
)

imagem = encode_image("dados\exemplo_grafico.jpg")

pergunta = "Descreva a imagem: "

mensagem = HumanMessage(
  content = [
    {
      "type" : "text", 
      "text" : pergunta
    },
    {
      "type" : "image_url",
      "image_url" : f"data:image/jpeg;base64,{imagem}"
    }
  ]
)

resposta = llm.invoke([mensagem])

print(resposta)