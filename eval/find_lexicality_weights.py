import os
import re
import pandas as pd
from pathlib import Path

project_dir = Path.cwd()

log_text = open(os.path.join(project_dir, "sbatch_logs/test_all-15782428_trained_with_excercise_tag_+system_prompt_topp0.9_topk50_temperature0.7.out")).read()

prompts = re.findall("<<\d+ Prompt>> ([\s\S]+?) <<Prompt>>", log_text)
generations = re.findall("<<\d+ Generation>> ([\s\S]+?) <<Generation>>", log_text)
len(prompts), len(generations)

generations = [one.replace("<exercise>", "").replace("</exercise>", "") for one in generations]

test_data = pd.read_csv(os.path.join(project_dir, "data/instruction_sets/instruct_test.csv"))
references = test_data.Output.tolist()

from lexicality.lexicality_eval import lexicality
result = lexicality(generations, references)
print("Original scores:", result)

# Calculate weights based on inverse square of scores
scores = list(result.values())
weights = [(1 - score)**2 for score in scores]
total_weight = sum(weights)
weights = [w/total_weight for w in weights]

# Calculate weighted final score
weighted_score = sum(score * weight for score, weight in zip(scores, weights))

print("\nWeights for each metric:")
for metric, weight in zip(result.keys(), weights):
    print(f"{metric}: {weight:.4f}")

print(f"\nFinal weighted score: {weighted_score:.4f}")
