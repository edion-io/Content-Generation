import re
import pandas as pd

def parse_params(text):
    text = text.strip()
    # Find content '(...)' or S or T or D or G or M or number (5 params)
    params= re.findall(r"\(.+?\)|\d+|S|T|D|G|M", text)
    params = [p.replace("(","").replace(")", "") for p in params]
    # assert len(params) == 5, "Parse result is not complete"
    assert len(params) == 5, text
    return params

with open("data/questions.txt", "r") as f:
    text = f.read()

text = "\n\n\n"+text
title_pattern = r"\n\n\n\(.+\).+\n"
# Find all titles (questions)
titles = re.findall(title_pattern, text)
# Find all contents (answers)
contents = re.split(title_pattern, text)
contents = [content.strip() for content in contents]
contents = contents[1:] if contents[0] == "" else contents

subject, exercise_type, difficulty, grade_level, modifier = ([] for _ in range(5))

for title in titles:
    params = parse_params(title)
    subject.append(params[0])
    exercise_type.append(params[1])
    difficulty.append(params[2])
    grade_level.append(params[3])
    modifier.append(params[4])

struct_data = pd.DataFrame({"subject": subject, "exercise_type": exercise_type, "difficulty": difficulty,
                             "grade_level": grade_level, "modifier": modifier, "content": contents})

sample_subjects_train = struct_data.groupby(by="subject").apply(lambda x: x.sample(n=50)).reset_index(drop=True)
sample_subjects_train.to_csv("data/sample_subjects_train.csv")

test_subjects = struct_data[~struct_data.index.isin(sample_subjects_train.index)]
sample_subjects_test = test_subjects.groupby(by="subject").apply(lambda x: x.sample(n=10)).reset_index(drop=True)
sample_subjects_test.to_csv("data/sample_subjects_test.csv")