# Copyright (C) 2024  Edion Management Systems
from openai import OpenAI
import glob
import json
import argparse
import re
import inflect

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
    if value == 'T':
        final = 'Give me an exercise'
    else:
        p = inflect.engine()
        value = value.split(',')
        # If multiple types are given, we phrase it as a combination of these types
        if len(value) > 1:
            final = 'Give me a combination of'
            for i, val in enumerate(value):
                if i != len(value) - 1:
                    final += f' {p.a(val)} {val.lower()},'
                else:
                    final += f' and {p.a(val)} {val.lower()}'
        # Otherwise we just ask for the single exercise type
        else:
            final = f'Give me {p.a(val)} {val.lower()}'

    return final

def grade_and_subject(grade: str, subject: str) -> str:
    # Formulate the "For a (grade) student" segment
    final = f'for a {f"{grade}{get_ordinal_suffix(int(grade))} grade " if grade != 'G' else ''}student'
    
    # Formulate the "learning (subject)" segment
    if subject == 'T':
        final += '.'
    else:
        final += f' learning {subject.lower()}.'
  
    return final

def modifier(value: list) -> str:
    

    if 'With Illustration With Answer With Material With Prerequisites With Context With Hint':
        pass

    # With Marks ()

    # With Instruction
    final = 'It should be '

    # Multi-part


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
    try:
        for mod in params[3].split(','):
            mod = mod.strip()
            if not (mod.startswith('Exercise') or mod.startswith('With') or mod.startswith('Activity') or mod == 'Multi-part' or mod == 'M'):
                print(mod, header)
    except:
        print(header)
        print(params)
    return
    
    # Formulate the initial instruction
    instruction = f"{exercise_type(params[1])} {grade_and_subject(params[2], params[0])}"
    # If there are any modifiers, formulate a new sentence describing them
    if params[3] != 'M':
        instruction += f' {modifier(params[3].split(','))}'

    return instruction


def make_instructions(file_path: str) -> list:
    """
    Preprocesses the questions.txt file to split by each header.

    Args:
        file_path (str): The path to the questions.txt file.

    Returns:
        list: A list of questions split by headers.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match headers
    header_pattern = r'\n(?=\(.*?\) \(.*?\))'
    questions = re.split(header_pattern, content)

    for q in questions:
        if q:
            params, body = q.split("\n", 1)
            instructionize(params)
    

    return questions

if __name__ == "__main__":
    make_instructions('data/questions.txt')

