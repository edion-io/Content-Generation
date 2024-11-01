import re
from openai import OpenAI
from utils import batch
import json
import copy

prompt = "I'm gonna give you text that needs to be put in latex format:\n1. Output the edited version of what I give you and only change what isn't in latex format. \n2. If you see lists that use letters, use the correct enumitem format,otherwise for lists with numbers, use regular \begin{enumerate}.\n3. If you see 'Answers' rewrite it as \subsection{Answers}\n4. Aside from 2, DO NOT add random subsections or random paragraphs. \n5. Any underscore needs to be written correctly (with a backslash before it). \n6. DO NOT output document class or packages, assume that we know what packages you use.\n7. DO NOT touch the first line (the header, which looks a little bit like: (French) (Matching) D G (With Answer) )"

if __name__ == "__main__":
    client = OpenAI(api_key="sk-proj-ltmkMm6qZ8oQCsusN5IOT3BlbkFJmsPopivPYwLtY7jlx5Pl")

    # Extract all the questions from the file
    with open("qs.txt", "r") as f:
        content = f.read()
    # Split the questions by header
    questions = re.split(r'(?m)(?=^\(Spanish\))', content)

    pattern = r'^\d+[\./]\s?.*$'
    final = []
    for i, q in enumerate(questions):
        if re.search(pattern, q, re.MULTILINE):
            lines = re.findall(pattern, q, re.MULTILINE)
            new = ["\\begin{enumerate}"]
            for line in lines:
                new.append(f"\\item{line[2:]}")
            new.append("\\end{enumerate}")
            q = q.replace("\n".join(lines), "\n".join(new)) 
        final.append(q)
    
    with open("new.txt", "w") as f:
        for q in final:
            f.write(q)
                    
    # for q in questions:
    #     q.rstrip()

    # # Batch the questions
    # batches, current_batch, current_tokens = [], [], 0
    # for i, q in enumerate(questions):
    #     current_tokens = batch(q, f"Spanish_{i}", current_tokens, current_batch, batches, prompt, True)
    
    # if current_batch:
    #     batches.append(current_batch)

    # # Save the smaller batches as separate files
    # for i, items in enumerate(batches):
    #     with open(f'tasks/batch_{i+1}.jsonl', 'w') as f:
    #         [f.write(json.dumps(item) + '\n') for item in items]