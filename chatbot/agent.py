# Copyright (C) 2024  Edion Management Systems
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


# Define parameters model.
class ExerciseParams(BaseModel):
    subject: str = Field(description="The subject of excercise")
    exercise_type: str = Field(description="Exercise type such as multiple choice question, short answer, fill-in-the-blank")
    grade: str = Field(description="grade level from 1 to 12  as per the American system")
    modification: str = Field(description="additional information that modify subject, exercise type, grade level.")


# Define generation function, where we can put in content generation LLM.
def generate_exercise(subject: str, exercise_type: str, grade: str, modification: str) -> str:
    """Generate exercise based on user's preference (4 parameters)"""
    return "Solve for x in the equation: 3x + 5 = 20. a) x = 3; b) x = 5; c) x = 10; d) x = 15"


# Define generator as a tool used by conversational LLM (chatbot).
exercise_generator = StructuredTool.from_function(
    func=generate_exercise,
    name="exercise_generator",
    description="Generate exercise based on user's preference (subject, excercise type, grade level, modification)",
    args_schema=ExerciseParams,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)


# Initiate chat model.
llm = ChatOllama(model="llama3.1")

# All possible tools.
tools = [exercise_generator]

# Define prompt template to specify system role & task, and the format of user input.
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""
        You are an exercise generation helper, you have the access to tools to serve for the user. Here is the design: your main task is to maintain conversation with the user; during this process, you are trying to identify 4 parameters from user’s input that is usefull for excercise generation, once you got the results you'll and present the generated exercise to the user right away. The user doesn’t know you are using tools.
        
        4 parameters needed to generate exercise:
        subject in {Math, Chemistry, Biology, Environmental Science, Physics, English, Foreign Languages, History/Social Studies, Music and Art and other possible subject 			taught in school}, required.
        excercise type in {multiple choice questions (MCQ), short answer form questions (SA), long answer form questions (LA), fill-in-the-blank (FB), Word Problems (WP), and so forth—any type of exercise one might encounter in class or a textbook}, optional.
        grade level in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} as per the American system, optional.
        modification: free text information that adding constraints to s, t, g. derived from conversation with the user, optional.

        Your response to the user should follow these principles: 
        1. Acting like an exercise generation assistant.
        2. You must try acquiring all the first 4 parameters (s, t, g, m) from the user in the conversation as much as possible.
        3. Only when the user provide all information they could, you can start call the tool. and present the results in a consistent way.
        4. You need to say you are preparing the excercise before calling the tool.        

        Follow these principles and be a good exercise generation assistant!
        """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# Bind tools to chat model
llm_with_tools = llm.bind_tools(tools)

# Wrap everything into an agent which can use tools and parse result
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# Define agent executor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Conversation loop with history
chat_history = []
while True:
    input_msg = input("Input: ")
    result = agent_executor.invoke({"input": input_msg, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=input_msg),
            AIMessage(content=result['output']),
        ]
    )
