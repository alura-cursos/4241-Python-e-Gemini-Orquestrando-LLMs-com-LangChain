from pydantic import BaseModel, Field
from typing import List

class DetalhesImagemModelo(BaseModel):
  titulo:str = Field(
    description="Defina o titulo adequado para a imagem que foi analisada."
  )
  descricao:str = Field(
    description="Coloque aqui uma descrição detalhada de sua análise para imagem."
  )
  rotulos:List[str] = Field(
    description="Defina três rótulos principais para a imagem analisada."
  )