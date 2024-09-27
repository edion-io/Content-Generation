from fastapi import FastAPI, HTTPException
import uvicorn
from echatbot import ExerciseChatbot
from definition import conversation_sys_prompt2

# Create the app
app = FastAPI()

# Load chatbot
chatbot = ExerciseChatbot(
    model_path="/home1/s4702328/chatbot/chat_exercise_Q4_K_M.gguf",
    system_prompt=conversation_sys_prompt2,
    device='gpu',
    temperature=0.2
)

# Add chat session
@app.get('/add_session')
def add_session():
    return {"session_id": chatbot.add_session()}

# Chat
@app.post('/chat')
def predict(user_input, session_id):
    if session_id not in chatbot.sessions:
        raise HTTPException(status_code=400, detail="session id not exist, please add session first or verify your session id")
    response = chatbot.run(user_input, session_id)
    return {'response': response}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)