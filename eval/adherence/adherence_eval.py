import re
import os
from openai import OpenAI
from dotenv import load_dotenv

# Set up your OpenAI API key
load_dotenv() 
assert os.getenv("OPENAI_API_KEY"), "OpenAI API key not found, please set it in envrioment variable OPENAI_API_KEY or in .env file"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt engineering
SYSTEM_PROMPT = """
Role:
You are an exercise generation evaluator. 

Task:
Given an query-answer pair, where the query is a demand and the answer is the generated exercise,
you will score the coherence of the excercise from 4 perspectives:
- Subject: score 1 if the subject is correct else 0.
- Exercise type: score 1 if the exercise type is correct else 0.
- Grade level: score 1 if the grade level is close to the given according to 1-12 Amecian grade system else 0.
- Modification: score 1 if the specific details matches the requirement else 0.
Don't be too strict in each perspective, you can give them score as long as it is fulfilled in a cetain level. Your final score is the sum of all these four scores.

Return:
Provide your answer in the following format:
{"Subject score": x, "Exercise type score": x, "Grade level score": x, "Modification score": x, "Total score": x}"""


def llm_call(sys_prompt, user_prompt):
    """
    LLM calling function: text generation using GPT-4o.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,  # Limit the number of tokens in the output
            temperature=0.7,  # Adjust the creativity level of the model
            n=1,  # Number of completions to generate
            stop=None  # Stop generation based on special characters or tokens
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def parse_response(response):
    """
    Parse llm result to dictionary
    """
    response = re.findall(r"{[\s\S]+}", response)[0]
    return eval(response)


def adherence_score(instruction: str, content: str):
    """
    Estimate adherence score.
    Args:
        intruction: user prompt, the request for excercise generation.
        content: the generated exercise as the response to instuction.
    """
    user_prompt = str({"Query": instruction, "Answer": content})
    response = llm_call(SYSTEM_PROMPT, user_prompt)
    try:
        result = parse_response(response)
        return result  # Return the full result dictionary, not just the total score
    except Exception as e:
        print(f"Error: {e}")
        print(f"LLM Response: {response}")
        return None