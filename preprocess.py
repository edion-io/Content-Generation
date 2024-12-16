# Copyright (C) 2024  Edion Management Systems
import argparse
import re
import inflect
from spellchecker import SpellChecker
import csv
from collections import defaultdict
from rapidfuzz import fuzz
import matplotlib.pyplot as plt

def plot_dist(labels: list, counts: list, title: str, xlab: str, path: str) -> None:
    # Plot the bar chart
    plt.bar(labels, counts, color='skyblue')

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel('Frequency')

    # Rotate x-axis labels
    plt.xticks(rotation=90, fontsize =10)
    # Fit the plot to the figure to avoid text overfitting
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'plots/{path}')

    # Clear the plot
    plt.clf()

def group_params(path: str) -> defaultdict:
    # Create a dictionary of parameters
    params = defaultdict(set)
    # Initialize a mapping for removed parameters to present parameters
    for k in 'TM':
        params[f'R_{k}'] = defaultdict(bool)
    # Initialize a counter for parameters
    for k in 'CSTGM':
        params[f'C_{k}'] = defaultdict(int)

    # Populate the dictionary of parameters while combining like terms
    for q in get_questions(path):
        # Reset the current combination of parameters
        combi = ''

        # Get the header of the question
        header = q.split("\n", 1)[0]

        # Extract the parameters from the header
        current_params = get_params(header)

        # Save each individual parameter
        for k, v in zip('STGM', current_params):
            if k in 'TM' and v not in ('T', 'M', None):
                for p in v.split(','):
                    # Strip the parameter of any spaces
                    p = p.strip()

                    # We check if the parameter has already been added or removed
                    if p not in params[k] and not params[f'R_{k}'][p]:
                        # Normalize the first parameter depending on whether we're dealing with exercise types or modifiers
                        if k == 'T':
                          if p not in 'ExerciseActivityQuestionGame':
                            cp1 = p.replace(' Exercise', "").replace(' Activity', "")
                        elif k == 'M':
                            cp1 = p.replace("Exercise ", "").replace("Activity ", "")
                        # Iterate through all saved parameters to check if there is any one too similar
                        for p2 in params[k]:
                            # Skip redundant cases
                            if k == 'T' and p2 in 'ExerciseActivityQuestionGame':
                                continue
                            else:
                                # Normalize the second parameter depending on whether we're dealing with exercise types or modifiers
                                if k == 'T':
                                    cp2 = p2.replace(' Exercise', "").replace(' Activity', "")
                                else:
                                    cp2 = p2.replace("Exercise ", "").replace("Activity ", "")

                                # Use fuzzy matching to obtain a similarity score
                                score = fuzz.token_sort_ratio(cp1, cp2)

                                # If the score is greater than 90 and parameters are truly similar then remove one
                                if score > 90 and not (('Exercise' in p and 'Activity' in p2) or ('Exercise' in p2 and 'Activity' in p)):
                                    params[f'R_{k}'][p] = p2

                        # If the parameter doesn't exist yet and there are no conflicts, add it instead
                        if not params[f'R_{k}'][p]:
                            params[k].add(p)

                    # Update the parameter combination and the count for this specific parameter
                    if params[f'R_{k}'][p]:
                        combi += params[f'R_{k}'][p]
                        params[f'C_{k}'][params[f'R_{k}'][p]] += 1
                    else:
                        combi += p    
                        params[f'C_{k}'][p] += 1 
            else:
                v = k if v is None else v.strip()
                params[k].add(v)
                # Update the parameter combination
                combi += v
                # Update the count for this specific parameter
                params[f'C_{k}'][v] += 1
            if k in 'STG':
                combi += '|'
        # Update the count for this combination of parameters
        params['C_C'][combi] += 1

    return params

def match_params(header: str):
    # Create a pattern for splitting
    pattern = r"""
    (\([^)]*\)|S)\s       # Capture text inside parentheses or "S"
    (\([^)]*\)|T)\s       # Capture text inside parentheses or "T"
    D\s                   # Match the literal "D"
    (\d+|G)\s             # Capture a number or "G"
    (\([^)]*\)|M)         # Capture text inside parentheses or "M"
    """
    return re.match(pattern, header, re.VERBOSE)

def get_params(header: str):
    # Split the parameters 
    match = match_params(header)

    # Clean the parameters
    return [p.replace('(', '').replace(')', '') if '(' in p else p for p in match.groups()]

def get_questions(path: str):
    # Open the file and get its contents
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match headers
    header_pattern = r'\n(?=\(.*?\) \(.*?\))'

    # Split the questions on their headers
    return re.split(header_pattern, content)

def get_ordinal_suffix(number: int):
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
    params = get_params(header)
    
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
    # Get the questions
    questions = get_questions(file_path)

    # Prepare the instructions list
    instructions = [('Input', 'Output')]
    for q in questions:
        if q.strip():
            # Split the question into header and body
            header, body = q.strip().split("\n", 1)

            # Split the header into parameters
            match = match_params(header)

            if match:
                # Check if the match consumed the entire header line
                if match.end() != len(header):
                    # Extra text detected after the header
                    print(f"Header with potential issue:\n'{header}'\n")
                # Proceed with instructionization
                instructions.append((instructionize(header), body.strip()))
            else:
                # Header does not match the expected pattern
                print(f"Header does not match expected pattern:\n'{header}'\n")
    return instructions

def check_header_spelling(file_path: str) -> None:
    """
    Reads the headers from the questions.txt file and checks for spelling mistakes.
    """
    # Initialize the spell checker
    spell = SpellChecker()

    # Get the questions
    questions = get_questions(file_path)

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
    # -------------------------
    # Parsers / Subparsers
    # -------------------------
    # Create the parser and subparsers
    argparser = argparse.ArgumentParser(description="Preprocess data for instruction-tuning.")
    subparsers = argparser.add_subparsers(dest="key", help="Subcommand description")

    parser_i = subparsers.add_parser("i", help="Augment a dataset to use it for instruction-tuning")
    parser_i.add_argument("-t", help="Keep the dataset in a .txt file, otherwise save it as a .csv", action="store_true")

    parser_v = subparsers.add_parser("v", help="Visualize the distributions of the inputs")

    # Parse the arguments
    args = argparser.parse_args()

    # -------------------------
    # Program logic
    # -------------------------
    if args.key == "i":
        # Turn the questions into an instruction tuning dataset
        questions = make_instructions('data/questions.txt')
        if args.t:
            # Write the new dataset to a text file
            with open('data/instructions.txt', 'w') as f:
                for q in questions:
                    f.write(q[0] + '\n' + q[1] + '\n\n\n')
        else:
            # Write the new dataset to a csv file
            with open('data/instructions.csv', mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(questions)
    elif args.key == 'v':
        params = group_params('data/questions.txt')

        # Print total number of unique parameters
        print('Number of subjects:', len(params['S']), 
              '\nNumber of exercise types:', len(params['T']), 
              '\nNumber of grades:', len(params['G']),
              '\nNumber of modifiers:', len(params['M']),
              '\nNumber of unique combinations:', len(params['C_C']))
        
        # Plot the distribution of the subjects
        s_s = sorted(params['C_S'].items(), key=lambda x: x[1], reverse=True)
        plot_dist([item[0] for item in s_s], [item[1] for item in s_s], 'Histogram of subject frequency', 'Subject', 'subject_dist.png')

        # Plot the distribution of the grades
        plot_dist(list(params['C_G'].keys()), list(params['C_G'].values()), 'Histogram of grade frequency', 'Grade', 'grade_dist.png')

        # Sort the dictionary by its values in descending order and take the top 10
        top_10 = sorted(params['C_T'].items(), key=lambda x: x[1], reverse=True)[:10]
        # Plot the distribution of the top 10 exercise types
        plot_dist([item[0] for item in top_10], [item[1] for item in top_10], 'Histogram of top 10 exercicse type frequency', 'Exercise Type', 'ex_type_dist.png')

        # Sort the dictionary by its values in descending order and take the top 10
        top_10 = sorted(params['C_M'].items(), key=lambda x: x[1], reverse=True)[:10]
        # Plot the distribution of the top 10 modifiers
        plot_dist([item[0] for item in top_10], [item[1] for item in top_10], 'Histogram of top 10 modifier frequency', 'Modifier', 'modifier_dist.png')

        # Sort the dictionary by its values in descending order and take the top 10
        top_10 = sorted(params['C_C'].items(), key=lambda x: x[1], reverse=True)[:10]
        # Plot the distribution of the top 10 combinations
        plot_dist([item[0] for item in top_10], [item[1] for item in top_10], 'Histogram of top 10 combination frequency', 'Combination', 'comb_dist.png')
        

    elif args.key == 's':
        pass