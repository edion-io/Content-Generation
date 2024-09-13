# Copyright (C) 2024  Edion Management Systems
from transformers import AutoTokenizer
import json
from prompt import conversation_sys_prompt2

def generate_exercise(subjects: tuple, exercise_types: tuple, grade_levels: tuple, modifications: tuple, ready: bool) -> float:
    """
    Generate excercise based on user preference. 
    
    Args:
        subjects: a tuple of exercise subjects for each exercise in order
        exercise_types: a tuple of exercise type for each exercise in order
        grade_levels: a tuple of grade levels for each exercise in order
        modifications: a tuple of modified information for each exercise in order
        ready: if the user give all paramters he can and its ready to generated exercise
    Returns:
        A list of exercises based on user's preference, as a string.
    """
    assert len(subjects) == len(exercise_types) == len(grade_levels) == len(modifications)
    if ready:
        return ("exercise 1", "exercise 2", "exercise 3")
    else:
        return "Not ready yet, ask the user for more information."

tools = [generate_exercise]

model_id = base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

with open("batch_request/clean_synthetic_conversations13+multi_params.json", 'r') as f:
    conversations = json.load(f)

data = []
for messages in conversations:
    # # With system prompt
    # msg = [{"role": "system", "content": conversation_sys_prompt2}] + messages['messages']
    # Without system prompt
    msg = messages["messages"]
    text = tokenizer.apply_chat_template(msg, tools=tools, tokenize=False, add_generation_prompt=False)
    data.append({'text': text})
    # print(text)
    # break

print("saving ...")
with open("batch_request/corrected_synthetic_conversations13+multi_params.json", 'w') as f:
    json.dump(data, f, indent=2)