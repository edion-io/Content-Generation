# Copyright (C) 2024  Edion Management Systems
import fitz
import glob
from utils import are_blocks_close, combine_blocks, has_imperative, is_near_top_or_bottom, find_first_number

# Specify the page range to process
PAGE_MIN = 5
PAGE_MAX = 99
# Specify the folder containing the PDF files
FOLDER = 'books/Computer Science/Primary'

if __name__ == "__main__":
    for i, path in enumerate(glob.glob(f"{FOLDER}/*.pdf")):
        grade = find_first_number(path)
        print(f"Processing {path}")
        # Open the PDF file
        pdf_document = fitz.open(path)

        # Save the subject
        subject = FOLDER.split('/')[1]

        # Iterate through all the pages
        for page_num in range(len(pdf_document)):
            # Load the page
            page = pdf_document.load_page(page_num)

            # Skip the first few pages (e.g., table of contents, glossary)
            if page_num < PAGE_MIN or page_num > PAGE_MAX:
                continue
            
            # Get page height
            page_height = page.rect.height

            # Get text with bounding box details
            text = page.get_text("dict")

            # Extract text blocks and their bounding boxes
            blocks = []
            for block in text["blocks"]:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        line_text = " ".join(span['text'] for span in line['spans'])
                        block_text += line_text + "\n"
                    
                    block_info = {'text': block_text, 'bbox': block['bbox']}
                    blocks.append(block_info)

            # Combine blocks that are close to each other
            combined_blocks = []
            while blocks:
                current_block = blocks.pop(0)
                group = [current_block]
                j = 0
                while j < len(blocks):
                    if are_blocks_close(current_block, blocks[j]):
                        group.append(blocks.pop(j))
                        current_block = combine_blocks(group)
                        j = 0  # Restart checking from the beginning
                    else:
                        j += 1
                combined_blocks.append(current_block)
            
            # Print combined blocks that contain imperative text near the top or bottom of the page (exercises)
            with open('questions.txt', 'a') as f:
                for combined_block in combined_blocks:
                    if has_imperative(combined_block['text']):
                        near_top, near_bottom = is_near_top_or_bottom(combined_block, page_height)
                        if near_top or near_bottom:
                            f.write(f"{subject} T D {grade} M\n")
                            f.write(f"{combined_block['text']}\n")

        # Close the PDF file
        pdf_document.close()