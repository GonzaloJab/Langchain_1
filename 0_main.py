import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    info = """Somos pioneros, somos innovación Empresa altamente tecnológica que diseña, 
    fabrica e implementa soluciones para el control de procesos y calidad. Está basada en 
    la tecnología de corrientes de Foucault e integrada con Inteligencia Artificial, Visión 
    Artificial y Análisis de Big Data (mediante redes neuronales y algoritmos) para la toma 
    de decisiones. Todo ello cuenta con herramientas de software que ponen la información de 
    forma fácil y accesible para los usuarios finales."""

    summary_template = """
    Resumen de la siguiente información: {info}
    1. Resumen de la información
    2. Un dato relevante de la información
    """

    summary_prompt = PromptTemplate.from_template(
        # input_variables=["info"], #No needed, declared in the template
        template=summary_template
    )

    llm = ChatCohere(model="command-r", temperature=0)
    # llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt | llm
    response = chain.invoke(input={"info": info})
    print(response.content)


if __name__ == "__main__":
    main()
