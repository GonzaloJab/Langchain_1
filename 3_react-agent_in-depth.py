from dotenv import load_dotenv
from langchain.agents import Tool, tool
from langchain_core.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
# Import Cohere and Ollama
from langchain_cohere import ChatCohere
from langchain_ollama import ChatOllama
from typing import Union, List

from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish

from langchain.agents.format_scratchpad import format_log_to_str


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters"""
    text = text.strip("'\n").strip("\"").strip("'")
    return len(text)

def find_tool_by_name(tools: List[Tool], name: str) -> Tool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name {name} not found")

if __name__ == "__main__":
    print("Hello, React Agent!")
    tools = [get_text_length]

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action,just the input
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])
        )
    # print(f"Prompt: {prompt}")
    # llm = ChatCohere(model="command-r", temperature=0, stop=["\nObservation","Observation:"])
    llm = ChatOllama(model="phi3.5:3.8b", temperature=0)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0, stop=["\nObservation","Observation:"])
    # format_log_to_str()

    intermediate_steps = []
    
    agent = ({"input": lambda x: x["input"],
             "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])} 
            | prompt 
            | llm 
            | ReActSingleInputOutputParser()
            )

    # ReAct Agent Loop
    question = "How many characters are in the word dog?"
    max_iterations = 4
    iteration = 0
    
    print("=== Starting ReAct Agent Loop ===")
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Input to agent: intermediate_steps = {intermediate_steps}")
        
        try:
            # Debug: Show formatted scratchpad
            if intermediate_steps:
                formatted_scratchpad = format_log_to_str(intermediate_steps)
                print(f"Formatted scratchpad:\n{formatted_scratchpad}")
            
            # Call the agent
            agent_step: Union[AgentAction, AgentFinish] = agent.invoke({
                "input": question,
                "agent_scratchpad": intermediate_steps
            })
            
            print(f"Agent step: {agent_step}")
            print(f"Type: {type(agent_step)}")
            
            # Check if agent is finished
            if isinstance(agent_step, AgentFinish):
                print(f"🎉 Agent finished!")
                print(f"Final Answer: {agent_step.return_values}")
                break
            
            # If agent wants to take an action
            elif isinstance(agent_step, AgentAction):
                tool_name = agent_step.tool
                tool_input = agent_step.tool_input
                
                print(f"🔧 Using tool: {tool_name}")
                print(f"Tool input: {tool_input}")
                
                # Find and execute the tool
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_input)
                
                print(f"📋 Observation: {observation}")
                
                # Add to intermediate steps
                intermediate_steps.append((agent_step, str(observation)))
                print(f"Updated intermediate_steps length: {len(intermediate_steps)}")
            
            else:
                print(f"❌ Unknown agent step type: {type(agent_step)}")
                break
                
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if iteration >= max_iterations:
        print(f"⚠️ Reached maximum iterations ({max_iterations}) without finishing")
    
    print("\n=== ReAct Agent Loop Complete ===")