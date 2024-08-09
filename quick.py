from utils import glob, fitz, find_first_number, get_images
def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def find_largest(x1, x2):
    if x1 > x2:
        return [x2, x1 - 1]
    else:
        return [x1, x2 - 1]
    
def main():
    numbers = {}
    for file in glob.glob("books/Math/Primary/Cambridge*.pdf"):
        num = find_first_number(file)
        numbers = []
        doc = fitz.open(file)
        for i in range(3, 5):
            page = doc.load_page(i)
            text = page.get_text("text")
            for line in text.split("\n"):
                if is_integer(line) and (len(numbers) == 0 or (len(numbers) > 0 and int(line) * 2 > numbers[-1])):
                    if int(line) not in numbers:
                        numbers.append(int(line))
        ranges = [find_largest(numbers[j], numbers[j+1]) for j in range(len(numbers) - 1)]
        get_images(file, ranges, 0, f"Math_{num}")
main()