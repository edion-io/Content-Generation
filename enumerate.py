import re
from openai import OpenAI
from utils import batch
import json
import copy

prompt = "You are a data annotator where each data is an exercise.  You will be shown exercises composed of a header and a body. The header is always the first line of the question. In each set of parentheses, everything should be separated with commas, not the word \"and\". For example :\n(Mathematics) (Number Statements, Fill in the Blank) D 6 (Multi-part)\n\nThe second set of parentheses of the header always encloses the exercise's type. It's okay for an exercise to have more than one type, however, the exercise types should always be very general (avoid specificity at all costs). For example, for questions about doing some kind of mathematical operation, you would write Arithmetic Exercise.  The last set of parentheses is the modifier. A modifier is a specification of what is written in any of the other sets of parentheses. For example, here it says Multi-part because the exercise this header belongs to has multiple questions. We want to keep the trend that the exercise type is general, while the modifier adds specificity. Your job is to follow these steps:\n\n1. Closely examine the header (specifically the exercise type and the modifier) and the question.\n2. Current exercise types may have types that are too specific. For example, Division Exercise is not good and we would prefer Arithmetic Exercise. When you find an exercise type that is too specific: \nI. Replace it with the more general type (unless it's already there). Exercise types should always end with \"Exercise\".\nII. Take the specific type and try to write it/them in the modifier in the format With ... or Exercise that has ... or Exercise with ... or Exercise on ...\n\nFor example:\n(Mathematics) (Percent and Division Exercise,  Writing and Explanation, Fill in the Blank) D 6 (Multi-part)\n\nwould become:\n(Mathematics) (Arithmetic Exercise, Short Answer, Fill in the Blank) D 6 (Exercise on percentages, Exercise with division, Multi-part)\n\nNOTE: You can replace/edit/remove exercise types but do not change modifiers that are already there, those are always correct.\n\n3. Output the entire new header with the body of the question"

if __name__ == "__main__":
    client = OpenAI(api_key="sk-proj-ltmkMm6qZ8oQCsusN5IOT3BlbkFJmsPopivPYwLtY7jlx5Pl")

    # Extract all the questions from the file
    with open("new2.txt", "r") as f:
        content = f.read()
    # Split the questions by header
    questions = re.split(r'(?m)(?=^\(Mathematics\))', content)

    for q in questions:
        q.rstrip()

    # Batch the questions
    batches, current_batch, current_tokens = [], [], 0
    for i, q in enumerate(questions):
        current_tokens = batch(q, f"Math_{i}", current_tokens, current_batch, batches, prompt, True)
    
    if current_batch:
        batches.append(current_batch)

    # Save the smaller batches as separate files
    for i, items in enumerate(batches):
        with open(f'tasks/batch_{i+1}.jsonl', 'w') as f:
            [f.write(json.dumps(item) + '\n') for item in items]