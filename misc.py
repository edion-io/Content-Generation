import fitz
from copy import deepcopy

FILE = 'books/German/German Language Learning This Book includes Learn German for Beginners, Phrase Book, Short Stories. Perfect For Travel Get... (Language Building Lab) (Z-Library).pdf'

def is_bold(font_name):
    return 'Bold' in font_name

def check_num(answer):
    i = 1
    while str(i) in answer:
        i += 1
    return i - 1

def check_period_and_letter(text, max_letter='K'):
    period_index = text.find('.')
    while period_index != -1:
        if period_index > 0 and text[period_index - 1].isalpha():
            letter_before_period = text[period_index - 1].upper()
            if letter_before_period <= max_letter:
                return True
        period_index = text.find('.', period_index + 1)
    return False

def get_unit_subunit(texts):
    last = texts.pop(0)
    unit, subunit = last.split('-')

    return int(unit), int(subunit), False if len(texts) > 0 else True

# Function to sort based on the split parts
def sort_key(s):
    parts = s.split('-')
    return int(parts[0]), int(parts[1])

if __name__ == "__main__":
    # Open the PDF file
    pdf_document = fitz.open(FILE)
    texts = {}
    exs = []
    collecting = False
    # Get all answers
    for page_num in range(40, 211):
        # Load the page
        page = pdf_document.load_page(page_num)
        blocks = page.get_text('dict')['blocks']
        for block in blocks:
            if 'lines' in block:
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text']
                        font_name = span['font']
                        bold = is_bold(font_name)
                        if page_num < 43:
                            print(exs, text)
                        if text.strip() == "Exercises" and bold:
                            exs.clear()
                            collecting = 1
                            continue
                        elif text.strip() == "Answers" and bold:
                            for x in exs:
                                texts[x] += '\nAnswers\n'
                            collecting = 2
                        elif collecting:
                            if bold:
                                if collecting in [1,2]:
                                    exercise = text
                                if 'Chapter' in text:
                                    collecting = 0
                                elif collecting == 1:
                                    exs.append(exercise)
                                if exercise not in texts:
                                    texts[exercise] = 'German T D G (With Answer)\n'
                            else:
                                texts[exercise] += text + '\n'
    with open("german.txt", 'w') as f:
        for val in texts.values():
            f.write(f'{val}\n\n')