from langchain.agents import AgentExecutor
from orquestrador import AgenteOrquestrador

def main():
  agente = AgenteOrquestrador()
  orquestrador = AgentExecutor(
    agent=agente.agente,
    tools=agente.tools,
    verbose=True
  )

  pergunta = "Faça uma análise da imagem exemplo_grafico.jpg"
  resposta = orquestrador.invoke({"input": pergunta})
  print(resposta)

if __name__ == "__main__":
  main()