import requests
import time
import concurrent.futures
import numpy as np
import random


API_URL = "http://127.0.0.1:8000"
USER_MSGS = [
    "Hello",
    "Math",
    "Multiple choice",
    "5th grade",
    "Nope",
    "Can I have two more similar"
]

def add_session():
    response = requests.get(f"{API_URL}/add_session")
    if response.status_code == 200:
        return response.json()["session_id"]
    else:
        raise Exception("Failed to add session")
    
def chat_request(session_id, user_input):
    response = requests.post(f"{API_URL}/chat", json={"user_input": user_input, "session_id": session_id})
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Chat request failed with status code {response.status_code}")
        
def chat_flow(session_id, response_times):
    for msg in USER_MSGS:
        start_time = time.time()
        try:
            print(chat_request(session_id, msg))
        except Exception as e:
            print(f"Request failed: {e}")
        end_time = time.time()
        print("Response time: ", end_time - start_time)
        response_times.append(end_time - start_time)
        # time.sleep(random.random()*5)

def pressure_test(num_session, asyn=True):
    if asyn:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(num_session):
                response_times = []
                session_id = add_session()
                print(f"Session ID: {session_id}")
                futures.append(executor.submit(chat_flow, session_id, response_times))
            concurrent.futures.wait(futures)

    else:
        for _ in range(num_session):
            response_times = []
            session_id = add_session()
            print(f"Session ID: {session_id}")
            chat_flow(session_id, response_times)
            average_reponse_time = np.mean(response_times)
            print(f"Average Response Time: {average_reponse_time:.4f} seconds")


if __name__ == "__main__":
    num_session = 5
    pressure_test(num_session=num_session, asyn=False)

    # while True:
    #     print("Hello world")
    #     time.sleep(1)