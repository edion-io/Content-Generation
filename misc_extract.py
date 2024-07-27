import fitz
import glob

def which_number(text):
    i = 1
    while f"{i}. " in text:
        i += 1
    return i - 1

if __name__ == "__main__":
    questions = []
    prompt = { 1 : ["All the letters of these words are jumbled up. Rearrange the letters so that they make meaningful words.", "Word Scramble"],
             0 : ["Guess the word in each case; some of the letters have been hidden by the underscore ( _ ) sign.", "Word Completion"]}
    i = 208
    incomplete = 0

    for path in glob.glob("books/Dutch/2300*.pdf"):
        doc = fitz.open(path)
        for page_num in range(doc.page_count):
            if page_num < 10:
                continue
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if incomplete == 0:
                p, t = prompt[i % 2]
                # Start with heading and prompt
                question = f'Dutch {t} D G (With Answer, With Hint)\n' + p + '\n'
                # Then add the exercises
                question += text[text.index("1."):].replace("_ ", "_")
                if which_number(question) != 5:
                    incomplete += 1
            elif incomplete == 1:
                question += text.replace("_ ", "_").replace("( ", "(").replace("* ", "*")
                incomplete += 1
            else:
                incomplete = 0
                ans_page = doc.load_page(i)
                ans_text = ans_page.get_text("text")
                question += text.replace("( ", "(").replace("* ", "*") + 'Answers\n' + ans_text[ans_text.index("1."):] + '\n'
                i += 1
                question = question.replace("4 .", "4.").replace("5 .", "5.")
                question = question.replace("H ints", "Hints")
                questions.append(question)
                if page_num == 207:
                    break
        doc.close()
    with open("questions.txt", "a") as f:
          f.writelines(questions)