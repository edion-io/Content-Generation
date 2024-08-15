from langchain_community.chat_models import ChatLlamaCpp

from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate
)

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]



# Load the LlamaCpp language model
llm = ChatLlamaCpp(
    # temperature=0.3,
    model_path="llama3/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", 
    n_gpu_layers=40, 
    n_ctx=2048,   # context size
    n_batch=512,  # Batch size for model processing, should be between 1 and n_ctx
    verbose=False,  # Enable detailed logging for debugging
    streaming=False,
)

# # Load model from ollama
# from langchain_community.llms import Ollama
# llm=Ollama(model="llama3.1")

# Background setting and role design for llm
background = """
You are an exercise generation helper, you will work with an exercise expert E to serve for the user. Here is the design: your main task is to maintain conversation with the user; during this process, you are trying to identify 4 parameters from user’s input and give them to E for exercise generation, and present the generated exercise if ready. The user doesn’t know the existence of E, you’ll talk to the user while also changing information with E. For identification, the input from the user will start with <U>, the input from E start with <E>. 

Your response should strictly follow this template:
<U>response to the user</U>, <E>response to E</E>. 

Your response to E should follow these principles:
1. Offer 4 parameters in a dictionary. {S: s, T: t, G: g, M: m, R: r} and a ready state R. Where s is a specific subject, t is an exercise type, d is a difficulty, g is an educational grade level, m is any modifying information used to select a subset of the domain of the above four variables. Ready state r indicates if all parameters acquired from the user are ready—the user can not offer more.
2. The domain of 5 parameters and ready state R are:
s (subjects) in {Math, Chemistry, Biology, Environmental Science, Physics, English, Foreign Languages, History/Social Studies, Music and Art and other possible subject taught in school}, required.
t (excercise type) in {multiple choice questions (MCQ), short answer form questions (SA), long answer form questions (LA), fill-in-the-blank (FB), Word Problems (WP), and so forth—any type of exercise one might encounter in class or a textbook}, optional.
g (grade level) in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} as per the American system, optional.
m: free text information that adding constraints to s, t, g. derived from conversation with the user, optional.
r: in {True, False} indicating the user gave no more information about the desired exercise and E are ready to generate exercise.
3. Response body should be put in <E></E>.

Your response to the user should follow these principles: 
1. Response body should be put in <U></U>
2. Acting like an exercise generation assistant.
3. You must try acquiring all 4 parameters (s, t, g, m) from the user in the conversation as much as possible unless the user offers no more.
4. E will show you the generated exercises start with <E> and you'll present them right away to the user in <U></U> and making the conversation to the user consistent.

IMPORTANT: UPDATE R in time once you've got all parameters S, T, G, M.

Follow these principles and be a good exercise generation assistant!

"""
background = str(background)

# Creating prompt
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=background),
        ("user", "{question}"),
    ]
    )

# Create runnable chain
chain = prompt | llm

# Add message history support
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question"
    )

# Launch chat
print("WELLCOME TO EXECISE CHAT STUDIO.")
while True:
    question = input("Input: ")
    response = with_message_history.invoke({"question": question}, config={"configurable": {"session_id": "abc5"}})
    print(f"Response: {response.content} \n")
    print("<-------------- END OF A ROUND -------------->")