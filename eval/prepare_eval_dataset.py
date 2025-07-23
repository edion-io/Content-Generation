import pandas as pd
import os
import sys
from pathlib import Path

# Add the root directory to Python path to import preprocess module
root_dir = Path(__file__).parent.parent
datasets_dir = root_dir.parent / "datasets"
sys.path.insert(0, str(root_dir))

from preprocess import get_questions, match_params, get_params

questions = get_questions(os.path.join(datasets_dir, "data/raw_datasets/test.txt"))
instructions = pd.read_csv(os.path.join(datasets_dir, "data/instruction_sets/instruct_test.csv"))

params_list = []

for question in questions:
    header, body = question.strip().split("\n", 1)

    # print(header)
    # print(body)

    match = match_params(header)
    if match:
        # Check if the match consumed the entire header line
        if match.end() != len(header):
            # Extra text detected after the header
            print(f"Header with potential issue:\n'{header}'\n")
        # Proceed with instructionization
        params = get_params(header)
        params_list.append(
            {
                "subject": params[0],
                "exercise_type": params[1],
                "grade_level": params[2],
                "modifier": params[3]            }
        )
        
    else:
        # Header does not match the expected pattern
        print(f"Header does not match expected pattern:\n'{header}'\n")

# Convert the list of parameter dictionaries into a DataFrame with the 4 specific columns
params_df = pd.DataFrame(params_list)

# Check if the row counts match before appending
if len(instructions) != len(params_df):
    print("Warning: The number of rows in 'instructions' and in 'params_list' do not match.")
else:
    # Reset indices to ensure proper alignment and concatenate the DataFrames column-wise
    instructions = pd.concat([instructions.reset_index(drop=True), 
                              params_df.reset_index(drop=True)], axis=1)
    print("New columns appended successfully. Updated 'instructions' shape:", instructions.shape)

instructions.to_csv(os.path.join(datasets_dir, "data/eval_datasets/eval_dataset.csv"), index=False)