# Copyright (C) 2024  Edion Management Systems
import argparse
import re
import inflect
from spellchecker import SpellChecker
import csv
from collections import defaultdict
from rapidfuzz import fuzz
import matplotlib.pyplot as plt

def get_removed(param: str, removed: dict) -> str:
    """ Verifies whether a parameter tag is an alias. Outputs the source if it is.

    Args:
        param (str): Either an exercise type or a question modifier.
        removed (dict): A dictionary mapping alias parameter tags to their source tags.

    Returns:
        str: The source tag if the parameter is an alias, otherwise the original parameter.
    """
    # If parameter is an alias we output the source, otherwise output the origin
    return removed[param] if removed[param] else param

def get_param(sample: str, param_idx: int, counts: dict = None, removed: dict = None, check_multi = False) -> str | bool:
    """ Either fetches a given parameter for sorting purposes or outputs a boolean indicating whether the parameter has multiple co-occuring levels.

    Args:
        sample (str): The sample being analyzed.
        param_idx (int): The index of the parameter if parameters exist in a list [S, T, G, M].
        counts (dict, optional): A dictionary of counts used to select the parameter with the least frequency of occurence. Defaults to None.
        removed (dict, optional): A dictionary mapping alias parameter tags to their source tags. Defaults to None.
        check_multi (bool, optional): Whether to output the parameter or to check for co-occuring class levels. Defaults to False.

    Returns:
        str | bool: The desired parameter or whether the parameter has co-occuring levels of its class.
    """
    single = True

    # Get the header of the question
    header = sample.split("\n", 1)[0]

    # Extract the parameter(s) from the header
    param = get_params(header)[param_idx]

    # Check for multi-parameter strings
    if param_idx in (1, 3) and removed:
        params = param.split(',')
        single = len(params) == 1
        c = 0
        # Choose the parameter with the lowest count
        for p in params:
            # Check if the parameter is an alias
            p = get_removed(p.strip(), removed)
            if not c:
                c = counts[p]

            if counts[p] <= c:
                param = p
    else:
        # Clean the parameter
        param = param.strip()

    return single if check_multi else param

def stratify_and_recombine(splits: list[list | dict], removed: dict, classify: tuple[bool, tuple[int, int]], n_counts: dict = None, ratios = [7,2,1]) -> list:
    """ Performs a stratified split for a list triplet (usually train/val/test), then recombines the 9 new splits back into a triplet.

    Args:
        splits (list[list | dict]): The specified list triplet (a list containing three lists or dictionaries).
        removed (dict): A dictionary mapping alias parameter tags to their source tags.
        classify (tuple[bool, tuple[int, int]]): Indicates whether the splits are list, which parameters to stratify by, and which parameter to sort the samples by.
        n_counts (dict, optional): A counter of the frequency of occurence of each class we are sorting the output splits by. Defaults to None.
        ratios (list, optional): The ratio used to allocate the samples to each split. Defaults to [7,2,1].

    Returns:
        list[list | dict]: Either a triplet of dicts containing lists of questions sorted by class, or lists containing questions.
    """
    new_splits = []

    # Stratify by a certain parameter
    for split in splits:
        new_splits.append(stratified_split(split, removed, n_counts, ratios=ratios, classify=classify))


    # Recombine the splits
    merged = new_splits[0]
    for sublist in new_splits[1:]:
        for idx, d in enumerate(sublist):
            if classify[0]:
                for key, value in d.items():
                    merged[idx][key].extend(value)
            else:
                merged[idx].extend(d)
    
    return merged

def choice_logic(list1: list, list2: list) -> str | None:
    """ Implements the logic for selecting a sample from two lists (unique and multi-class).
    List1 gets priority over list2.

    Args:
        list1 (list): The primary list which we want to draw from.
        list2 (list): The secondary list.

    Returns:
        str | None: A sample or None if both lists are empty. 
    """
    if len(list1) != 0:
        return list1.pop()
    elif len(list2) != 0:
        return list2.pop()
    else: 
        return None

def update_class(questions: list, counts: dict, param_idx: tuple[int, int], splits: list,
                ratios: list, removed: dict = None, max_samples = 10, skip = False) -> None:
    """ Allocates samples to a train, validation and test set using a stratification algorithm.
    The train set has priority over the remainder of sets and also primarily obtains "unique" samples.
    The validation set has priority over the test set and both primarily obtain multi-class samples.

    Args:
        questions (list): A list of questions containing questions pertaining to the class 'label'.
        counts (dict): A dictionary of frequency of occurrence for the new class we are sorting the stratified set by.
        param_idx (tuple[int, int]): The index position of the desired parameter we are counting while stratifying.
        splits (list): A list containing the train, validation and test splits.
        ratios (list): A list representing the ratio which we should follow in allocating samples to each respective set.
        removed (dict, optional): A mapping of removed class labels to their new label. Defaults to None.
        max_samples (int, optional): The maximum number of samples that we will allocate in this function call. Defaults to 10.
        skip (bool, optional): Whether to treat the splits as dictionaries (False) or as lists (True). Defaults to False.
    """

    # Initialize a list to hold the different samples
    multi, unique = [], []

    # Split the questions on a basis of whether they have a unique or multi-class label
    for q in questions: 

        # Get the parameter used for stratification
        single = get_param(q, param_idx[0], counts, removed, True)
            
        # Update the sub classes accordingly
        (unique if single else multi).append(q)

    # Initialize a 2^14 bit integer to keep track of the number of samples in each split
    count = 0

    for _ in range(len(questions)):
        
        # Check which split has priority
        idx = find_idx(max_samples, count, ratios)
  
        # Select a sample based on the split we are allocating to
        # Val/Test will always be idx 1 or 2
        # Train will always be idx 0
        # Val/Test sets prioritize multi-class parameters; Train set prioritizes unique
        q = choice_logic(multi, unique) if idx else choice_logic(unique, multi)

        # If q is None, we've run out of samples so we stop updating the split
        if q == None:
            return q
        
        # Update the counter for the class count in each split
        count += (1 << (14 * idx))

        if skip:
            # Update the split without paying attention to class 
            splits[idx].append(q)
        else:
            # Get the parameter used for classification
            param = get_param(q, param_idx[1], counts, removed)

            splits[idx][param].append(q)

def find_idx(max_samples: int, splits: int | list, ratios: list) -> int:
    """ Returns the index of the split that has the highest priority of allocation (based on the current percentages of samples allocated to each split).

    Args:
        max_samples (int): The maximum number of samples that we will allocate in this function call.
        splits (int | list): A bit-representation of the count of this class in each set or a list containing the containers for each split.
        ratios (list): A list representing the ratio which we should follow in allocating samples to each respective set.

    Returns:
        int: The index of the split which is next in line for allocation.
    """
    # Add all ratios together
    total = sum(ratios)

    # Compute the percentages of samples allocated to each split relative to their ideal allocations
    if type(splits) == int:
        percentages = [((splits >> (14 * i)) & 16383) * (total / (max_samples * r)) for i, r in enumerate(ratios)]
    else:
        percentages = [len(s) * (total / (max_samples * r)) for r, s in zip(ratios, splits)]

    # Find the split with the smallest percentage allocation (relative to its ideal allocation)
    # Splits get higher priority based on their positions
    idx = 0
    lowest = percentages[idx]
    for i, p in enumerate(percentages[1:]):
        if p < lowest:
            lowest = p
            idx = i + 1
    return idx
    

def stratified_split(questions: dict, removed: dict = None, n_counts: dict = None, n_splits = 3, ratios = [6, 2, 2], classify: tuple[bool, tuple[int, int]] = (False, None)) -> list:
    """ Performs a stratified split on a dataset.

    Args:
        questions (dict): A dictionary mapping labels to subsets of questions.
        removed (dict): A dictionary mapping parameter names to their general parameter assigned after grouping.
        f_counts (dict): A dictionary of counts for the next parameter we will use to stratify.
        n_splits (int, optional): The total number of splits to perform. Defaults to 3.
        ratios (list, optional): A list representing the ratio which we should follow in allocating samples to each respective set.. Defaults to [7, 2, 1].
        classify (tuple[bool, tuple[int, int]], optional): A tuple determining whether there will be more classes to stratify and the current class to stratify. Defaults to (False, None).

    Returns:
        list: A list containing the individual splits obtained post-stratification.
    """
    # Sort the data from smallest to largest count of occurence
    sorted_data = sorted(questions.items(), key = lambda x: len(x[1]))

    # Initialize lists for each split
    if classify[0]:
        splits = [defaultdict(list) for _ in range(n_splits)]
    else:
        splits = [[] for _ in range(n_splits)]

    # Cycle through each class and the counts of samples per class
    for item in sorted_data:

        # Count the amount of samples per class we can allocate to each split
        leftover = len(item[1]) % 10

        # Compute the max number of samples divisible by 10
        max_samples = len(item[1]) - leftover

        if len(item[1]) >= 10:
            # Partition the maximum amount of samples of this class that is divisible by 10
            update_class(questions[item[0]][:max_samples], n_counts, classify[1], splits, ratios, removed, max_samples, skip = not classify[0])

        # Add the leftover samples if there are any
        if leftover:
            update_class(questions[item[0]][-leftover:], n_counts, classify[1], splits, ratios, removed,skip = not classify[0])
        
    return splits

def plot_dist_top_n(d: dict, title: str, xlab: str, path: str, n: int = False) -> None:
    """ Plots the frequency distribution of the top "n" levels of a given class for a dataset.

    Args:
        d (dict): A dictionary mapping the class levels to their frequency of occurence.
        title (str): The title of the plot.
        xlab (str): The x-label.
        path (str): The path used to save the plot.
        n (int, optional): The cutoff point for the plot (top "n" levels). If false, takes the length of the dictionary. Defaults to False.
    """
    n = n if n else len(d)
    # Sort the dictionary based on the values
    items = sorted(d.items(), key = lambda x: x[1], reverse = True)[:n]
    # Extract the counts and labels separately
    counts = [item[1] for item in items]
    labels = [item[0] for item in items]
    # Plot the histogram of the data
    plot_dist(labels, counts, title, xlab, path)


def plot_dist(labels: list, counts: list, title: str, xlab: str, path: str) -> None:
    """ Plots the frequency distribution for all levels of a given class in a dataset.

    Args:
        labels (list): A list of all the levels of the class.
        counts (list): A list of all the counts of each level of the class, in the same order as the labels.
        title (str): The title of the plot.
        xlab (str): The x-label.
        path (str): The path used to save the plot.
    """
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

def group_params(path: str, keep_qs = False) -> defaultdict:
    """ Computes counts of all the classes and the combinations of classes in a given dataset. 
    Uses fuzzy matching to group like-levels of a class together and represent them under a single source tag.
    Optionaly creates a dictionary sorting the samples of the dataset by subject (S).
    

    Args:
        path (str): The path of the dataset whose classes we are analyzing/sorting/grouping.
        keep_qs (bool, optional): Whether to create a dictionary sorting the samples of the dataset by subject (S). Defaults to False.

    Returns:
        defaultdict: A dictionary containing mappings of class aliases to their source tags, counters of all classes and their combinations,
          and possibly a dictionary sorting the samples of the dataset by subject (S).
    """
    # Create a dictionary of parameters
    params = defaultdict(set)
    # Initialize a mapping for removed parameters to present parameters
    for k in 'TM':
        params[f'R_{k}'] = defaultdict(bool)
    # Initialize a counter for parameters
    for k in 'CSTGM':
        params[f'C_{k}'] = defaultdict(int)
    # Initialize a dict to classify questions by subject
    params[f'Q_S'] = defaultdict(list)

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
                # Save the questions by subject if requested
                if keep_qs and k == 'S':
                    params[f'Q_{k}'][v].append(q)
            if k in 'STG':
                combi += '|'
        # Update the count for this combination of parameters
        params['C_C'][combi] += 1

    return params

def match_params(header: str) -> re.Match:
    """ Extracts the parameters (S, T, G, M) from a question header (the input sample).

    Args:
        header (str): The header of a question containing all of its parameters.

    Returns:
        re.Match: A regex match object containing the parameters of the question.
    """
    # Create a pattern for splitting
    pattern = r"""
    (\([^)]*\)|S)\s       # Capture text inside parentheses or "S"
    (\([^)]*\)|T)\s       # Capture text inside parentheses or "T"
    D\s                   # Match the literal "D"
    (\d+|G)\s             # Capture a number or "G"
    (\([^)]*\)|M)         # Capture text inside parentheses or "M"
    """
    return re.match(pattern, header, re.VERBOSE)

def get_params(header: str) -> list[str]:
    """ Extracts the parameters of a given question header, cleans them and outputs them.

    Args:
        header (str): The specified question header.

    Returns:
        list[str]: A list of parameters, in the order (S, T, G, M).
    """
    # Split the parameters 
    match = match_params(header)

    # Clean the parameters
    return [p.replace('(', '').replace(')', '') if '(' in p else p for p in match.groups()]

def get_questions(path: str) -> list[str]:
    """ Fetches questions from a given .txt file and splits them by header.

    Args:
        path (str): The path to the specified .txt file.

    Returns:
        list[str]: A list of questions.
    """
    # Open the file and get its contents
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match headers
    header_pattern = r'\n(?=\(.*?\) \(.*?\))'

    # Split the questions on their headers
    return re.split(header_pattern, content)

def get_ordinal_suffix(number: int) -> str:
    """ Computes the ordinal suffix for a given number.

    Args:
        number (int): The specified number.

    Returns:
        str: The ordinal suffix used for the corresponding number.
    """
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
    """ Converts a given exercise type (T) parameter into text instructions for LLMs.

    Args:
        value (str): The specified exercise type.

    Returns:
        str: Text instructions that can be used for instruction tuning.
    """
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
    """ Converts a given grade (G) and subject (S) parameter into text instructions for LLMs.

    Args:
        grade (str): The specified grade parameter.
        subject (str): The specified subject parameter. 

    Returns:
        str: Text instructions that can be used for instruction tuning.
    """
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
    """ Converts a given list of modifier (M) parameters into text instructions for LLMs.

    Args:
        modifiers (list): The specified list of modifier parameters.

    Returns:
        str: Text instructions that can be used for instruction tuning.
    """
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
    """ Converts a given question header (the input [S, T, G, M]) into a set of text instructions that can be used for training LLMs.

    Args:
        header (str): The specified question header.

    Returns:
        str: Text instructions that can be used for instruction tuning.
    """
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
    parser_i.add_argument("path", help="The path fo the file contaiing the dataset.")
    parser_i.add_argument("-t", help="Keep the dataset in a .txt file, otherwise save it as a .csv", action="store_true")

    parser_v = subparsers.add_parser("v", help="Visualize the distributions of the inputs")
    parser_v.add_argument("-p", help="Plot the frequency distributions of the data's classes",action="store_true")

    parser_s = subparsers.add_parser("s", help="Perform a stratified split on a dataset to obtain training, validation and test subsets.")
    parser_s.add_argument("ratio", help="The desired ratio for the set's train/val/test split, in the format 'x,y,z'")

    # Parse the arguments
    args = argparser.parse_args()

    # -------------------------
    # Program logic
    # -------------------------
    
    # Manually augments a dataset for instruction-tuning.
    if args.key == "i":
        # Turn the questions into an instruction tuning dataset
        questions = make_instructions(f'data/{args.path}.txt')
        if args.t:
            # Write the new dataset to a text file
            with open(f'data/instruct_{args.path}.txt', 'w') as f:
                for q in questions:
                    f.write(q[0] + '\n' + q[1] + '\n\n\n')
        else:
            # Write the new dataset to a csv file
            with open(f'data/instruct_{args.path}.csv', mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(questions)

    # Performs data exploration
    elif args.key == 'v':
        params = group_params('data/questions.txt')

        # Print total number of unique parameters
        print('Number of subjects:', len(params['S']), 
              '\nNumber of exercise types:', len(params['T']), 
              '\nNumber of grades:', len(params['G']),
              '\nNumber of modifiers:', len(params['M']),
              '\nNumber of unique combinations:', len(params['C_C']))
        
        if args.p:
            # Plot the distribution of the subjects
            plot_dist_top_n(params['C_S'], 'Histogram of subject frequency', 'Subject', 'subject_dist.png')

            # Plot the distribution of the grades
            plot_dist(list(params['C_G'].keys()), list(params['C_G'].values()), 'Histogram of grade frequency', 'Grade', 'grade_dist.png')

            # Plot the distribution of the top 10 exercise types
            plot_dist_top_n(params['C_T'], 'Histogram of top 10 exercise type frequency', 'Exercise Type', 'ex_type_dist.png', 10)
            
            # Plot the distribution of the top 10 modifiers
            plot_dist_top_n(params['C_M'], 'Histogram of top 10 modifier frequency', 'Modifier', 'modifier_dist.png', 10)

            # Plot the distribution of the top 10 combinations
            plot_dist_top_n(params['C_C'], 'Histogram of top 10 combination frequency', 'Combination', 'comb_dist.png', 10)
        
        sorted_top = sorted(params['C_M'].items(), key=lambda x: x[1], reverse=True)
        top_n = [item[1] for item in sorted_top if item[1] > 2]
        print(len(top_n))
        print(f"top {len(top_n)} modifiers account for {100 * sum(top_n)/sum(x[1] for x in params['C_M'].items())}% of data")

        sorted_top = sorted(params['C_T'].items(), key=lambda x: x[1], reverse=True)
        top_n = [item[1] for item in sorted_top if item[1] > 2]
        print(len(top_n))
        print(f"top {len(top_n)} account for {100 * sum(top_n)/sum(x[1] for x in params['C_T'].items())}% of data")

    # Performs a stratified multi-dimensional split
    elif args.key == 's':
        # Parse the desired ratios
        if args.ratio:
            ratios = [int(r) for r in args.ratio.split(',')]

        # Stratified sampling of dataset
        params = group_params('data/questions.txt', True)

        # STEP 1: Stratify by Subject and partition by grade labels
        splits = stratified_split(params['Q_S'], classify=(True, (0, 2)), ratios=ratios)

        # STEP 2-4: Stratify by Grade and partition by exercise type labels
        # then, stratify by Exercise Type and partition by modifier labels
        # and finally stratify by modifier
        for removed, classify, counter in zip((params['R_T'], params['R_M'], params['R_M']),
                                              ((True, (2, 1)),  (True, (1, 3)),  (False, (3, 3))),
                                              (params['C_T'], params['C_M'], params['C_M'])):
            splits = stratify_and_recombine(splits, removed, classify, counter, ratios)

        # Compute the total number of samples.
        total = sum(len(s) for s in splits)
        # Compute and output the percentage of the dataset that each split represents.
        print(f"Final set composition:\n\nTrain: {len(splits[0]) * 100 / total:.2f}%\nVal: {len(splits[1]) * 100 / total:.2f}%\nTest: {len(splits[2]) * 100 / total:.2f}%")

        # Save the subsets into .txt files
        for split, label in zip(splits, ('train', 'val', 'test')):
            with open(f'data/{label}.txt', 'w') as f:
                f.write("\n\n\n".join(split))