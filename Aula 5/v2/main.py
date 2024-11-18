from langchain.agents import AgentExecutor
from orquestrador import AgenteOrquestrador


pergunta = "Me explique o quesão desvios condicionais?"

agente = AgenteOrquestrador()
executor = AgentExecutor(agent=agente.agente,
                        tools=agente.tools,
                        verbose=True)
resposta = executor.invoke({"input" : pergunta})
print(resposta.get("output"))
