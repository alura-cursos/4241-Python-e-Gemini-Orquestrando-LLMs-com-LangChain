from langchain.tools import BaseTool

class FerramentaAnalisadoraImagem(BaseTool):
  name:str = "FerramentaAnalisadoraImagem"
  description:str = """
  Utilize esta ferramenta sempre que for solicitado que você faça uma análise
  de imagem.

  # Entradas Requiridas
  - 'nome_imagem' (str) : Nome da imagem a ser analisada com extensão de JPG. Exemplo: teste.jpg ou teste.jpeg
  """
  return_direct : bool = False

  def _run(self, acao):
    return ""