from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from my_models import GEMINI_PRO, GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate


imagem = encode_image("Aula 2/v1/dados/exemplo_grafico.jpg")

llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_PRO)

template_analise_imagens = ChatPromptTemplate.from_messages(
    [
        (
          "system",  
          """Assuma que você é um analisador de imagens experiente, e que sua tarefa é criar rótulos 
          inteligentes que ajudem a descrever e a encontrar imagens salvas.
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
