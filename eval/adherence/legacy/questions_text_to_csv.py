import pandas as pd
import os
import sys
from pathlib import Path

# Add the root directory to Python path to import preprocess module
root_dir = Path(__file__).parent.parent.parent
datasets_dir = root_dir.parent / "datasets"
sys.path.insert(0, str(root_dir))

from preprocess import get_questions, match_params, get_params


questions = get_questions(os.path.join(datasets_dir, "data/raw_datasets/questions.txt"))

# Create dataframe and save to csv file
data = []

for question in questions:
    header, body = question.strip().split("\n", 1)
    match = match_params(header)
    if match:
        params = get_params(header)
    else:
        print(f"Header does not match expected pattern:\n'{header}'\n")
        continue
    data.append(
        {
            "subject": params[0],
            "exercise_type": params[1],
            "grade_level": params[2],
            "modifier": params[3],
            "content": body
        }
    )

struct_data = pd.DataFrame(data)
struct_data.to_csv(os.path.join(datasets_dir, "data/raw_datasets/questions.csv"), index=False)