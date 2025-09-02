import os
from dotenv import load_dotenv
# Import, PromptTemplate, OllamaEmbeddings, PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
# Import ChatOllama
from langchain_ollama import ChatOllama

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("Retrieving data...")
    embeddings = OllamaEmbeddings(model="phi3.5:3.8b")
    llm = ChatOllama(model="phi3.5:3.8b", temperature=0)

    query = "What LLM is recommended to use in agents?"
    # chain = PromptTemplate.from_template(query) | llm
    # response = chain.invoke(input={})
    # print(response.content)

    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings
        )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrieval_chain = create_retrieval_chain(
                                             retriever=vectorstore.as_retriever(),
                                             combine_docs_chain=combine_docs_chain
                                             )
    
    # response = retrieval_chain.invoke(input={"input": query})
    
    # print(response)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,
    just say that you don't know. Use three sentences maximum and keep the answer concise.
    
    {context}

    Question: {question}

    Answer:
    """
    custom_rag_prompt = PromptTemplate.from_template(template=template, input_variables=["context", "question"])

    rag_chain = (
                {"context": vectorstore.as_retriever() | format_docs , "question": RunnablePassthrough()} |
                custom_rag_prompt | 
                llm
                )
    
    response = rag_chain.invoke(input={"question": query})
    print(response.content)