# Copyright (C) 2024  Edion Management Systems
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import conversation_sys_prompt2



base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
new_model = "llama3-8b-chat-exercise/checkpoint-1047-3epoch"

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
)

# Merge adapter with base model
model = PeftModel.from_pretrained(base_model_reload, new_model)
model = model.merge_and_unload()


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
    max_len = max(len(subjects), len(exercise_types), len(grade_levels), len(modifications))
    is_equal_len = (len(subjects) == len(exercise_types) == len(grade_levels) == len(modifications))
    if not is_equal_len:
        pass
        # TODO - expand to make them equal length.
    if ready:
        if "Math" in str(subjects):
            return ["What is 3/4 of 12? A)6 B) 9 C) 12 D) 15"] * max_len
        if "English" in str(subjects):
            return ["What habitat do tigers prefer? a) Desert b) Jungle c) Tundra d) Ocean"] * max_len
        if "Biology" in str(subjects):
            return ["Plants perform photosynthesis to make their own food. True or False?"] * max_len
    else:
        return "Not ready yet, ask the user for more information."

tools = [generate_exercise]

# Initial input and response
messages = [
  {"role": "system", "content": conversation_sys_prompt2},
  {"role": "user", "content": "<U> Hi, I need some Math excercise </U>"}
]

inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(out[0][len(inputs["input_ids"][0]):])
print(response)
messages.append({"role": "assistant", "content": response})
print("-------------------------------")

# Conversation loop
while True:
    user_input = input("Input: ")
    messages.append({'role': 'user', 'content': user_input})
    inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(out[0][len(inputs["input_ids"][0]):])
    print(response)
    if '"name": "generate_exercise"' in str(response):
        # Extract parameters
        response = re.findall(r"\{.*\}", response)[0]
        tool_call = eval(response.replace("true", "True").replace("false", "False").replace("null", "None"))
        params = tool_call["parameters"]
        tool_call = {"name": "generate_exercise", "arguments": params}
        messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
        messages.append({"role": "tool", "name": "generate_exercise", "content": generate_exercise(**params)})
        # Respond based on tool calling result
        inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        # print(tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=False))
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=128)
        response = tokenizer.decode(out[0][len(inputs["input_ids"][0]):])
        print(response)
    messages.append({"role": "assistant", "content": response})
    print("-------------------------------")