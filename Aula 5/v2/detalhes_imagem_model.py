from pydantic import BaseModel, Field
from typing import List

class DetalhesImagemModelo(BaseModel):
    titulo: str = Field(
        description="Defina um título para a imagem que será analisada."
    )
    descricao: str = Field(
        description="Coloque aqui uma descrição detalhada da imagem que analisada."
    )
    rotulos: List[str] = Field(
        description="Defina três rótulos que representem a imagem e facilitem sua identificação."
    )