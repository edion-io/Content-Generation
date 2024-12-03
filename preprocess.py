# Copyright (C) 2024  Edion Management Systems
import argparse
import re
import inflect
from spellchecker import SpellChecker

# -------------------------
# Parsers / Subparsers
# -------------------------
# Create the parser and subparsers
argparser = argparse.ArgumentParser(description="Preprocess data for instruction-tuning.")

def get_ordinal_suffix(number):
    # Handle special cases for 'teen' numbers
    if 11 <= number % 100 <= 13:
        return "th"
    # Determine the suffix based on the last digit
    last_digit = number % 10
    if last_digit == 1:
        return "st"
    elif last_digit == 2:
        return "nd"
    elif last_digit == 3:
        return "rd"
    else:
        return "th"

def exercise_type(value: str) -> str:
    # If no exercise type is specified, we simply ask for an exercise
    if value in ('T', None):
        final = 'Give me an exercise'
    else:
        p = inflect.engine()
        value = value.split(',')
        # If multiple types are given, we phrase it as a combination of these types
        if len(value) > 1:
            final = 'Give me a combination of'
            for i, val in enumerate(value):
                if i != len(value) - 1:
                    final += f' {p.a(val.lower()).strip()}'
                    if len(value) > 2:
                        final += ','
                else:
                    final += f' and {p.a(val.lower()).strip()}'
        # Otherwise we just ask for the single exercise type
        else:
            final = f'Give me {p.a(value[0].lower()).strip()}'

    return final

def grade_and_subject(grade: str, subject: str) -> str:
    # Formulate the "For a (grade) student" segment
    final = f'for a {f"{grade}{get_ordinal_suffix(int(grade))} grade " if grade not in ('G', None) else ''}student'
    
    # Formulate the "learning (subject)" segment
    if subject in ('S', None):
        final += '.'
    else:
        subject = subject.lower() if subject not in 'French English Dutch Spanish German' else subject
        final += f' learning {subject}.'
  
    return final

def modifier(modifiers: list) -> str:
    # Initialize parameters
    w_mods, remaining, final = [], [], ''
    p = inflect.engine()

    # Separate the "With ..." modifiers first
    for modifier in modifiers:
        if modifier in 'With Answer With Material With Prerequisites With Context With Hint':
            w_mods.append(modifier.strip())
        else:
            remaining.append(modifier.strip())

    # If there are any "With ..." modifiers, then create a sentence with them
    if w_mods:
        for i, modifier in enumerate(w_mods):
            mod = modifier.replace('With ', '').lower()
            if mod != 'context':
                mod += 's'
            if i == 0:
                final += f' It should have a subsection for {mod}'
            elif i == len(w_mods) - 1:
                final += f', and {mod}'
            else:
                final += f', {mod}'

            # Separate the period-adding logic for cases where len(w_mods) == 1
            if i == len(w_mods) - 1:
                final += '.'

    # Handle remaining modifiers
    for modifier in remaining:
        if modifier == 'With Illustration':
          final += ' The exercise should contain some kind of a graphic component (illustration, diagram, table, etc).'
        elif modifier == 'With Marks':
            final += ' There should be marks/points assigned to the question.'
        elif modifier == 'With Instruction':
            final += ' The exercise should have a top sentence that serve as instructions.'
        elif modifier == 'Multi-part':
            final += ' It should have multiple steps.'
        else:
            split = modifier.split(' ')
            # If the second word is with, we replace it with involving
            if split[1] == 'with':
                modifier = modifier.replace('with', 'involving', 1)
            # Put the first word to lowercase
            modifier = modifier.replace(split[0], split[0].lower())
            final += f' It should be {p.a(modifier)}.'
    
    return final


def instructionize(header: str) -> str:
    # Create a pattern for matching parameter groups
    pattern = r"""
    (\([^)]*\)|S)\s       # Capture text inside parentheses or "S"
    (\([^)]*\)|T)\s       # Capture text inside parentheses or "T"
    D\s                     # Match the literal "D"
    (\d+|G)\s               # Capture a number or "G"
    (\([^)]*\)|M)         # Capture text inside parentheses or "M"
    """
    # Split the parameters
    match = re.match(pattern, header, re.VERBOSE)
    params = match.groups()

    # Clean the parameters
    params = [p.replace('(', '').replace(')', '') if '(' in p else p for p in params]
    
    # Formulate the initial instruction
    instruction = f"{exercise_type(params[1])} {grade_and_subject(params[2], params[0])}"
    
    # If there are any modifiers, formulate a new sentence describing them
    if params[3] not in ('M', None):
        instruction += f'{modifier(params[3].split(','))}'

    return instruction


def make_instructions(file_path: str) -> list:
    """
    Preprocesses the questions.txt file to split by each header.

    Args:
        file_path (str): The path to the questions.txt file.

    Returns:
        list: A list of questions split by headers.
    """
    # Open the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match headers
    header_pattern = r'\n(?=\(.*?\) \(.*?\))'
    # Split the questions on their headers
    questions = re.split(header_pattern, content)

    # Prepare the instructions list
    instructions = []
    for q in questions:
        if q.strip():
            # Split into header and body
            if "\n" in q:
                params, body = q.strip().split("\n", 1)
            else:
                params = q.strip()
                body = ''
            # Use your existing pattern
            pattern = r"""
            (\([^)]*\)|S)\s       # Capture text inside parentheses or "S"
            (\([^)]*\)|T)\s       # Capture text inside parentheses or "T"
            D\s                   # Match the literal "D"
            (\d+|G)\s             # Capture a number or "G"
            (\([^)]*\)|M)         # Capture text inside parentheses or "M"
            """
            match = re.match(pattern, params, re.VERBOSE)
            if match:
                # Check if the match consumed the entire header line
                if match.end() != len(params):
                    # Extra text detected after the header
                    print(f"Header with potential issue:\n'{params}'\n")
                # Proceed with instructionization
                instructions.append(instructionize(params) + '\n' + body.strip())
            else:
                # Header does not match the expected pattern
                print(f"Header does not match expected pattern:\n'{params}'\n")
    return instructions

# New function to check spelling in headers
def check_header_spelling(file_path: str) -> None:
    """
    Reads the headers from the questions.txt file and checks for spelling mistakes.
    """
    spell = SpellChecker()
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match headers
    header_pattern = r'\n(?=\(.*?\) \(.*?\))'
    # Split the questions on their headers
    questions = re.split(header_pattern, content)

    for q in questions:
        if q.strip():
            params = q.strip().split("\n", 1)[0]
            # Extract words from the header
            header_text = params.replace('(', '').replace(')', '')
            words = re.findall(r'\b\w+\b', header_text)
            misspelled = spell.unknown(words)
            if misspelled:
                print(f"Spelling mistakes in header: '{params}'")
                print("Misspelled words:", ', '.join(misspelled))
                print()

if __name__ == "__main__":
    
    

    questions = make_instructions('data/questions.txt')

    with open('data/instructions.txt', 'w') as f:
        for q in questions:
            f.write(q + '\n\n\n')