import re
import os
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal, Optional

# Set up your OpenAI API key
load_dotenv() 
assert os.getenv("OPENAI_API_KEY"), "OpenAI API key not found, please set it in envrioment variable OPENAI_API_KEY or in .env file"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AdherenceScore(BaseModel):
    subject_score: Optional[int]
    exercise_type_score: Optional[int]
    grade_level_score: Optional[int]
    modifier_score: Optional[int]


# Prompt engineering
SYSTEM_PROMPT = """
Role:
You are an exercise generation evaluator. 

Task:
Given an query-answer pair, where the query is a demand and the answer is the generated exercise,
you will score the adherence of the excercise from 4 perspectives:
- Subject: Exercise subject, can be math, reading, writing, etc. Score 1 if the subject is correct else 0.
- Exercise type: Define exercise type/format, e.g. multiple choice, fill in the blank, etc. Score 0-4, 0 is completely not adherent, 4 is completely adherent.
- Grade level: Education level of the students in American school system 0-12. Score 0-4, 0 is completely not adherent, 4 is completely adherent.
- Modifier: Additional instruction for the exercise for specific requirements, e.g. "It should question about algebra", etc. Score 0-4, 0 is completely not adherent, 4 is completely adherent.

Note: If the query does not mention any of the subject, exercise type, grade level, or modifier, set the score to null.
    For example:
    - Query: "Generate a fill in the blank math exercise for grade 3 students, "
    - Answer: "1 + 1 = __"
    - return: {
        "subject_score": 1,
        "exercise_type_score": 4,
        "grade_level_score": 1,
        "modifier_score": None
    }

    - Query: "'Give me a multiple choice for a 1st grade student learning arts. It should be an exercise on symmetrical shapes.'"
    - Answer: "'Draw 2 shapes that are asymmetrical"
    - return: {
        "subject_score": 1,
        "exercise_type_score": 0,
        "grade_level_score": 1,
        "modifier_score": 3
    }

    - Query: "Give me a short answer exercise for a student learning computer science. It should have multiple steps. It should be an exercise on class diagrams."
    - Answer: "Who proposed the theory of evolution?"
    - return: {
        "subject_score": 0,
        "exercise_type_score": 0,
        "grade_level_score": None,
        "modifier_score": 0
    }

Return your response using the structured format provided.
{"subject_score": x, "exercise_type_score": x, "grade_level_score": x, "modifier_score": x}
"""


def llm_call(sys_prompt, user_prompt, method: Literal["structured output", "prompt parsing"] = "structured output"):
    """
    LLM calling function: text generation using GPT-4o with structured output.
    """
    try:
        if method == "structured output":
            print("Using structured output")
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=AdherenceScore,
                max_tokens=150,  # Limit the number of tokens in the output
                    temperature=0.7,  # Adjust the creativity level of the model
            )
            return response.choices[0].message.parsed
    
        elif method == "prompt parsing":
            print("Using prompt parsing")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,  # Limit the number of tokens in the output
                temperature=0.7,  # Adjust the creativity level of the model
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

def adherence_score(instruction: str, content: str, method: Literal["structured output", "prompt parsing"] = "structured output"):
    """
    Estimate adherence score.
    Args:
        intruction: user prompt, the request for excercise generation.
        content: the generated exercise as the response to instuction.
        method: "structured output" or "prompt parsing"
    """
    user_prompt = str({"Query": instruction, "Answer": content})
    response = llm_call(SYSTEM_PROMPT, user_prompt, method)
    try:
        if response:
            if method == "structured output":
                result = response.model_dump()
            elif method == "prompt parsing":
                result = parse_response(response)
            print(result)
            return result  # Return the full result dictionary
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        print(f"LLM Response: {response}")
        return None
    

if __name__ == "__main__":
    result = adherence_score(
        "Generate a math exercise for grade 3 students",
        "Who is the president of the United States?", 
        method="prompt parsing"
    )
    print(result)
    print("-"*100)
    result = adherence_score(
        "Generate a fill in the blank math exercise for grade 5 students, it should be an exercise about addition",
        "1 + 1 = __", 
        method="prompt parsing"
    )
    print(result)