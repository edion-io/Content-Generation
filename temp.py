from openai import OpenAI
import re
from utils import batch
import json

prompt = "We want to format diagram descriptions so that a model can use it to generate diagrams for exercises. A diagram description uses the format:\n[STRDGM] text [STPDGM]\nWhen provided with an exercise, ensure that the exercise's diagram(s) adhere to each of these rules:\n1. If a diagram contains multiple shapes or objects and they aren't supposed to be part of one unified image, then split the diagram.\ne.g.\ninput:\n[STRDGM] A circle, a triangle, and a square [STPDGM]\noutput:\n [STRDGM] A circle [STPDGM]\\n [STRDGM] A triangle [STPDGM]\\n [STRDGM] A square [STPDGM]\n2. Remove instances of \"a situational image\" or \"an image\" or anything that is similar to that.\ne.g.\ninput:\n[STRDGM] A situational image of a circle [STPDGM]\noutput:\n[STRDGM] A circle [STPDGM]\n3. Provide the necessary detail so that the generated diagram is sufficient for a student to do the exercise. Do not keep unnecessary information if it isn't helpful to the machine for drawing and/or won't lead to a loss of important info for the student viewing the image. For example, if the exercise asks one to find the area of a triangle: \ninput:\n[STRDGM] A triangle [STPDGM]\noutput:\n[STRDGM] A triangle labeled with base 4cm and height 6cm [STPDGM]\n4. Replace general terms like \"shape\", \"object\" or \"item\" with specific concrete terms that describe the particular instance you are referring to.\ne.g.\ninput:\n[STRDGM]  A grid with squares that form a whole shape. The whole shape is 15x15. [STPDGM] \noutput:\n[STRDGM] A grid with stacked-up squares that form a rectangle. The whole rectangle is 15x15. [STPDGM]\n5. If a diagram describes a table that is already written in Latex in the body of the exercise, the description is unnecessary and must be removed.  \n6. If a diagram describes a blank space or any space for writing answers, it is also unnecessary and must be removed. \n7. A diagram should always be positioned above the subsection{Answers} (if there is one). \ne.g.\ninput:\n\\subsection{Answers}\n[STRDGM] text [STPDGM]\noutput:\n[STRDGM] text [STPDGM]\n\\subsection{Answers}\n8. If a diagram makes an exercise way easier by defeating the purpose of the exercise, remove it.\n\nOnce you are done applying as many of the above as you can, then output the resulting question, maintaining the same format.\n\nExample with multiple changes:\ninput:\n(Mathematics) (Drawing Exercise, Fraction Exercise) D 5 (With Illustration)\n\nThis picture represents \\( \\frac{1}{3} \\) of a shape. It has 5 rectangles. Draw the whole shape. Can you find another way of doing it?\n\n[STRDGM] A situational image of a grid with 5 rectangles representing \\( \\frac{1}{3} \\) of a whole shape. The whole shape should be represented as 15 rectangles in total. [STPDGM]\n\noutput:\n(Mathematics) (Drawing Exercise, Fraction Exercise) D 5 (With Illustration)\n\nThis picture represents \\( \\frac{1}{3} \\) of a shape. It has 5 rectangles. Draw the whole shape. Can you find another way of doing it?\n\n[STRDGM] A grid with 5 rectangles representing \\( \\frac{1}{3} \\) of a whole triangle. [STPDGM]"

existing = []
if __name__ == "__main__" :
    client = OpenAI(api_key="sk-proj-ltmkMm6qZ8oQCsusN5IOT3BlbkFJmsPopivPYwLtY7jlx5Pl")

    # Extract all the questions from the file
    with open("q.txt", "r") as f:
        content = f.read()

    # Split the questions by header
    questions = re.split(r'(?m)(?=^\(English\))', content)

    indexes = []

    for i, q in enumerate(questions):
        added = False
        if "With Answer" in q or "\\subsection{Answers}" in q:
            # Separate the header and the question
            sep = q.split("\n", 1)
            header = sep[0]
            question = sep[1].strip()
            h_bool = "With Answer" in header
            q_bool = "\\subsection{Answers}" in question

            if h_bool and not q_bool:
                header_e = re.split(r'\s+(?=(?:[^()]*\([^()]*\))*[^()]*$)', header)
                mod = header_e.pop()
                if mod == "(With Answer)":
                    mod = "M"
                else:
                    last_e = mod[1:-1]
                    last_e = [s.strip() for s in last_e.split(",")]
                    idx = last_e.index("With Answer")
                    last_e.pop(idx)
                    if len(last_e) == 1:
                        mod = f"({last_e[0]})"
                    else:
                        mod = "(" + ", ".join(last_e) + ")"
                header_e.append(mod)
                added = True
            elif q_bool and not h_bool:
                header_e = re.split(r'\s+(?=(?:[^()]*\([^()]*\))*[^()]*$)', header)
                mod = header_e.pop()
                if mod == "M":
                    mod = "(With Answer)"
                else:
                    mod = "(" + mod[:-1] + ", With Answer)"
                header_e.append(mod)
                added = True

        if "With Illustration" in q or "[STRDGM]" in q:
            # Separate the header and the question
            edge = False
            sep = q.split("\n", 1)
            header = sep[0]
            question = sep[1].strip()
            h_bool = "With Illustration" in header
            q_bool = "[STRDGM]" in question
            if h_bool and not q_bool:
                if not added:
                    header_e = re.split(r'\s+(?=(?:[^()]*\([^()]*\))*[^()]*$)', header)
                mod = header_e.pop()
                if "With Illustration" not in mod:
                    header_e.append(mod)
                    mod = header_e.pop(1)
                    edge = True
                if mod == "(With Illustration)":
                    mod = "M" if not edge else "E"
                else:
                    last_e = mod[1:-1]
                    last_e = [s.strip() for s in last_e.split(",")]
                    idx = last_e.index("With Illustration")
                    last_e.pop(idx)
                    if len(last_e) == 1:
                        mod = f"({last_e[0]})"
                    else:
                        mod = "(" + ", ".join(last_e) + ")"
                if edge:
                    header_e.insert(1, mod) 
                else:
                    header_e.append(mod)
                added = True
            elif q_bool and not h_bool:
                if not added:
                    header_e = re.split(r'\s+(?=(?:[^()]*\([^()]*\))*[^()]*$)', header)
                mod = header_e.pop()
                if mod == "M":
                    mod = "(With Illustration)"
                else:
                    mod = "(" + mod[:-1] + ", With Illustration)"
                header_e.append(mod)
                added = True
        if added:
            indexes.append((i, " ".join(header_e) + "\n" + question))
      
    for idx in indexes:
        questions[idx[0]] = idx[1]
    
    # Write the modified questions to a file
    with open("new2.txt", "w") as f:
        f.write("\n\n\n".join(questions))

            
    # # Batch the questions
    # batches, current_batch, current_tokens = [], [], 0
    # for i, q in enumerate(questions):
    #     current_tokens = batch(q, f"Math_{i}", current_tokens, current_batch, batches, prompt, True)
    
    # if current_batch:
    #     batches.append(current_batch)

    # # Save the smaller batches as separate files
    # for i, items in enumerate(batches):
    #     with open(f'tasks/batch_{i+1}.jsonl', 'w') as f:
    #         [f.write(json.dumps(item) + '\n') for item in items]
