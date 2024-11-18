from langchain.agents import AgentExecutor
from orquestrador import AgenteOrquestrador


pergunta = "Faça uma análise da imagem exemplo_grafico.jpg"

agente = AgenteOrquestrador()
executor = AgentExecutor(agent=agente.agente,
                        tools=agente.tools,
                        verbose=True)
resposta = executor.invoke({"input" : pergunta})
print(resposta.get("output"))
