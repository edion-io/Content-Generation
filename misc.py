import fitz
from copy import deepcopy

FILE = 'books/German/The Everything Essential German Book All you need to Learn German in no time (Edward Swick) (Z-Library).pdf'

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

    unit = 2
    subunit = 0
    collecting = 0
    texts = {}
    answer = None

    # Get all answers
    for page_num in range(314, len(pdf_document)):
        # Load the page
        page = pdf_document.load_page(page_num)

        # Iterate through the text's
        text = page.get_text("text")
        for line in text.split('\n'):
            if "Chapter" in line:
                continue
            if f'{unit}-{subunit + 1}' == line.replace("Exercise", "").strip() or f'{unit + 1}-1' == line.replace("Exercise", "").strip() or collecting == 10:
                if answer is not None:
                    texts[f'{unit}-{subunit}'] = answer
                if line.replace("Exercise", "").strip() == f'{unit + 1}-1':
                    subunit = 1
                    unit += 1
                else:
                    subunit += 1
                collecting = 1
            elif collecting == 1:
                answer = line + '\n'
                collecting += 1
            elif collecting == 2:
                answer += line + '\n'
                if subunit == 2 and unit == 19 and "said Father cannot work" in line:
                    collecting = 10

    keys = sorted(deepcopy(list(texts.keys())), key=sort_key)
    unit, subunit, done = get_unit_subunit(keys)
    max_num = check_num(texts[f'{unit}-{subunit}'])
    question = None
    collecting = 0
    final = False

    for page_num in range(33, 313):
        # Load the page
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        for i, line in enumerate(text.split('\n')):
            if f"{unit}-{subunit}" in line.strip() or (final and f'{max_num}.' in line):
                if (final and f'{max_num}.' in line):
                    question += line + '\n'
                if question is not None:
                    texts[f'{p_unit}-{p_subunit}'] = f'German T D G (With Answer)\n{question}\nAnswers\n{texts[f'{p_unit}-{p_subunit}']}'
                p_unit, p_subunit = unit, subunit
                if final:
                    break
                if done:
                    final = True
                else:
                    unit, subunit, done = get_unit_subunit(keys)
                collecting = 1
                max_num = check_num(texts[f'{p_unit}-{p_subunit}'])
            elif collecting == 1:
                question = line + '\n'
                collecting = 2
            elif collecting == 2:
                question += line + '\n'
                if f'{max_num}.' in line:
                    collecting = 0
    print(texts)
    with open("german.txt", 'w') as f:
        for val in texts.values():
            f.write(f'{val}\n\n')