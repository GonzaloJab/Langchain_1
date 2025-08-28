from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_cohere import ChatCohere
# from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

tools = [TavilySearch()]
# llm = ChatOllama(model="gemma3:270m")
llm = ChatCohere(model="command-r", temperature=0)

react_prompt = hub.pull("hwchase17/react")

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "tool_names", "agent_scratchpad", "format_instructions"],
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
chain = agent_executor


def main():
    response = chain.invoke({"input": "Which day is today?"})
    print(response)


if __name__ == "__main__":
    main()
