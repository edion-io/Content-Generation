import re
import fitz  # PyMuPDF

def extract_german_textbook(pdf_path, output_path, start_page=179, end_page=240):
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Extract text from specified pages
    full_text = ""
    for page_num in range(start_page - 1, min(end_page, doc.page_count)):
        full_text += doc[page_num].get_text()
    
    # Define patterns for content, questions, and answers
    chapter_pattern = r'Chapter.*?\n(.*?)Zusammenfassung'
    questions_pattern = r'Fragen(.*?)Questions'
    answers_pattern = r'Antworten(.*?)Answers'
    
    # Find all matches
    chapters = re.findall(chapter_pattern, full_text, re.DOTALL)
    questions = re.findall(questions_pattern, full_text, re.DOTALL)
    answers = re.findall(answers_pattern, full_text, re.DOTALL)
    
    print(f"Found {len(chapters)} chapters, {len(questions)} question sections, and {len(answers)} answer sections.")
    
    # Process and combine sections
    formatted_sections = []
    for i in range(max(len(chapters), len(questions), len(answers))):
        section = "(German Language) (Reading Comprehension) D G (With Answer)\n\n"
        
        if i < len(chapters):
            # Merge chapter content into a single paragraph
            chapter_content = ' '.join(line.strip() for line in chapters[i].split('\n') if line.strip())
            section += chapter_content + "\n\n"
        
        if i < len(questions):
            question_content = questions[i].strip()
            section += question_content + "\n\n"
        
        if i < len(answers):
            answer_content = answers[i].strip()
            # Replace "Antwoord" with "Answers"
            answer_content = answer_content.replace("Antwoord", "Answers")
            section += answer_content
        
        formatted_sections.append(section)
    
    # Write all formatted sections to a single file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n' + '='*50 + '\n\n'.join(formatted_sections))
    
    print(f"Extraction and formatting complete. {len(formatted_sections)} sections saved to {output_path}")

# Use the function
extract_german_textbook("/home/arkadiy/Documents/Edion/german.pdf", "/home/arkadiy/Documents/Edion/germanstories.txt")