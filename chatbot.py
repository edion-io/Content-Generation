from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate
)

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Ollama

import re


class ExerciseChatbot():
    def __init__(self, model_name: str = "llama3.1", system_prompt: str = "") -> None:
        self.store = {}
        self.load_model(model_name)
        self.set_system_prompt(system_prompt)
        self.chain = self.prompt | self.model
        self.run_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question"
            )
        self.session_id = "abc5"

    @classmethod
    def load_model(self, model_name: str) -> Ollama:
        self.model = Ollama(model=model_name)

    def set_system_prompt(self, system_prompt: str = "") -> ChatPromptTemplate:
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                ("user", "{question}"),
            ]
            )
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    def parse_response(self, response: str) -> tuple[str, dict]:
        # Define patterns
        user_pattern = r"<U>(.*?)</U>"
        params_pattern = r"<E>(.*?)</E>"

        # Extracting the response to user
        user_match = re.search(user_pattern, response)
        user_response = user_match.group(1) if user_match else ""

        # Extracting the parameters
        params_match = re.search(params_pattern, response)
        parameters = params_match.group(1) if params_match else ""

        parameters = eval(parameters)

        assert type(parameters) == dict
        assert all(key in parameters.keys() for key in ['S', 'T', 'G', 'M'])

        return user_response, parameters
    
    def run_one_round(self, input: str):
        # Get response considering history
        response = self.run_with_history.invoke({"question": input},
                        config={"configurable": {"session_id": self.session_id}})
        print(response+"\n")
        # Parse based on the designed pattern
        to_user, parameters = self.parse_response(response)
        return to_user, parameters
    
    def run(self):
        parameters = {'R': False}
        while True:
            question = input("Input: ")
            if not parameters['R']:
                to_user, parameters = self.run_one_round(question)
                yield to_user
            else:
                exercise = self.generate_exercise(parameters)
                print(f"Delivering excercise: {exercise}")
                to_user, parameters = self.run_one_round(exercise)
                yield to_user

    def generate_exercise(parameters: dict) -> str:
        exercise = """
                    What is the correct shape of a figure with 4 sides and 4 right angles?
                    A) Square
                    B) Rectangle
                    C) Triangle
                    D) Rhombus</Math Exercise>
                """
        exercise = "<E> I'm the expert, present the following generated excercise to the user: " + exercise
        return exercise


if __name__ == "__main__":

    system_prompt = """
        You are an exercise generation helper, you will work with an exercise expert E to serve for the user. Here is the design: your main task is to maintain conversation with the user; during this process, you are trying to identify 4 parameters from user’s input and give them to E for exercise generation, and present the generated exercise if ready. The user doesn’t know the existence of E, you’ll talk to the user while also changing information with E. For identification, the input from the user will start with <U>, the input from E start with <E>. 

        Your response should strictly follow this template:
        <U>response to the user</U>, <E>response to E</E>. 

        Your response to E should follow these principles:
        1. Offer 5 parameters in a dictionary. {'S': s, 'T': t, 'G': g, 'M': m, 'R': r}. Where s is a specific subject, t is an exercise type, g is an educational grade level, m is any modifying information used to select a subset of the domain of the above four variables. Ready state r indicates if all parameters acquired from the user are ready—the user can not offer more.
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
        3. The initial parameters dictionary is {'S': None, 'T': None, 'G': None, 'M': None, 'R': False}
        3. You must try acquiring all the first 4 parameters (s, t, g, m) from the user in the conversation as much as possible and update them gradually. Till the user offers no more, then set Ready state r=Ture.
        4. E will show you the generated exercises start with <E> and you'll present them right away to the user in <U></U> and making the conversation to the user consistent.
        5. After presenting the excercise to the user, reset 'R': False in the parameters dictionary.

        Follow these principles and be a good exercise generation assistant!
    """

    chatbot = ExerciseChatbot(model_name="llama3.1", system_prompt=system_prompt)
    for answer in chatbot.run():
        print(answer)
        print("<-------------- END OF A ROUND -------------->")
