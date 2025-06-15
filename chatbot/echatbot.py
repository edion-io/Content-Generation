# Copyright (C) 2024  Edion Management Systems
import re
from llama_cpp import Llama
from definition import tools, generate_exercise
import uuid


class ExerciseChatbot():
    def __init__(self, model_path:str, system_prompt:str, device:str, temperature:float =0.2) -> None:
        self.device = device
        self.temperature = temperature
        self.sessions = {}
        self.model = self.load_model(model_path)
        self.system_prompt = system_prompt

    def load_model(self, model_path: str) -> Llama:
        n_gpu_layers = 0 if self.device == "cpu" else -1
        return Llama(model_path=model_path,
                           n_ctx=4096,
                           n_threads=6,
                           n_gpu_layers=n_gpu_layers)
    
    def add_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = [{"role": "system", "content": self.system_prompt}]
        return session_id
    
    def remove_session(self, session_id):
        del self.sessions[session_id]

    def get_session(self, session_id):
        return self.sessions[session_id]
    
    def run(self, user_input, session_id):
        # Get session chat history
        messages = self.get_session(session_id)
        messages.append({'role': 'user', 'content': user_input})
        # Generate response
        response = self.model.create_chat_completion(messages=messages, tools=tools, max_tokens=256, temperature=self.temperature)
        response = response["choices"][0]["message"]
        # If tool call, process and update response with exercise
        if '"name": "generate_exercise"' in str(response):
            # Filter and parse parameters
            content = re.findall(r"\{.*\}", response["content"])[0] 
            content = eval(content.replace("true", "True").replace("false", "False").replace("null", "None"))
            params = content["parameters"]
            # Add tool calling with template to chat history
            tool_call = {"name": "generate_exercise", "arguments": params}
            messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
            # Add tool call result (exercise) to chat history
            messages.append({"role": "tool", "name": "generate_exercise", "content": generate_exercise(**params)})
            # Get the final response to user
            response = self.model.create_chat_completion(messages=messages, tools=tools, max_tokens=256, temperature=self.temperature)
            response = response["choices"][0]["message"]
        # Add response to chat history
        messages.append(response)
        # Update sessions
        self.sessions[session_id] = messages
        return response


if __name__ == "__main__":

    from definition import conversation_sys_prompt2
    model_path = "/home1/s4702328/chatbot/chat_exercise_Q4_K_M.gguf"

    chatbot = ExerciseChatbot(model_path=model_path, system_prompt=conversation_sys_prompt2, device='gpu', temperature=0.2)
    session_id = chatbot.add_session()

    while True:
        user_input = input("Input: ")
        response = chatbot.run(user_input, session_id)
        print(response)
        print("<-------------- END OF A ROUND -------------->")