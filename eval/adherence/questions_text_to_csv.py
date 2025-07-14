import re
import pandas as pd


def parse_params(text):
    """
    Parse `(Computer Science) (Practical Exercise, Activity) D 2 M` like text to 5 parameters
    Subject, Exercise type, Difficulty, Grade level, Modifier
    """
    text = text.strip()
    # Find content '(...)' or S or T or D or G or M or number (5 params)
    params= re.findall(r"\(.+?\)|\d+|S|T|D|G|M", text)
    params = [p.replace("(","").replace(")", "") for p in params]
    # assert len(params) == 5, "Parse result is not complete"
    assert len(params) == 5, text
    return params

# Read original text file
with open("data/raw_datasets/questions.txt", "r") as f:
    text = f.read()

# Split parameters head and corresponding exercise
title_pattern = r"\([a-zA-Z].+\).+ D .+\n"
titles = re.findall(title_pattern, text)   # Find all titles (questions)
contents = re.split(title_pattern, text)   # Find all contents (answers)
contents = [content.strip() for content in contents]
contents = contents[1:] if contents[0] == "" else contents

# Create dataframe and save to csv file
data = []

for i, title in enumerate(titles):
    params = parse_params(title)
    data.append(
        {
            "subject": params[0],
            "exercise_type": params[1],
            "difficulty": params[2],
            "grade_level": params[3],
            "modifier": params[4],
            "content": contents[i]
        }
    )

struct_data = pd.DataFrame(data)
struct_data.to_csv("data/raw_datasets/questions.csv", index=False)