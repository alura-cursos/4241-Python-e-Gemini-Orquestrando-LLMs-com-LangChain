from langchain.tools import BaseTool

class FerramentaAnaliseImagem(BaseTool):
    name : str = "FerramentaAnaliseImagem"
    description : str = """
    Utilize esta ferramenta sempre que for solicitado que você faça uma análise de imagem. 
    
    # Entradas Requeridas
    - 'nome_imagem' (str) : Nome da imagem a ser analisada com extensão em JPG. Exemplo teste.jpg ou teste.jpeg
    """
    return_direct : bool = False

    def _run(self, acao):
        

        return ""