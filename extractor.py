# Copyright (C) 2024  Edion Management Systems
from openai import OpenAI
from utils import upload_image, batch_items, submit_batch, extract_raw_questions, PDF_to_images
import glob
import json
import argparse
import re
from utils import SUBJECTS

# -------------------------
# Parameters
# -------------------------
# Specify the folder/file containing any relevant files
PATH = 'german.txt'
# Specify the folder containing the images
IMAGE_FOLDER = 'imgs'
# Specify the batch folder
BATCH_FOLDER = 'tasks'
# Specify the question header (for extracting raw questions from text)
HEADER = ""
# Specify the Cloudinary client ID and secret
CLOUDINARY_API = "667797151493891"
CLOUDINARY_SECRET = "WuKdiXBzcwzUgOsdOey5J9E8k7c"

# Specify the prompt for the chat completion task
PROMPT = "You are a data annotator where each data is an exercise.  You will be shown exercises composed of a header and a body. The header is always the first line of the question. In each set of parentheses, everything should be separated with commas, not the word \"and\". For example :\n(French) (Grammar Exercise, Fill in the Blank) D 6 (Multi-part, Exercise on adjectives)\n\nThe second set of parentheses of the header always encloses the exercise's type. It's okay for an exercise to have more than one type, however, the exercise types should always be very general (avoid specificity at all costs). For example, for questions about adjectives you would write Grammar Exercise.  The last set of parentheses is the modifier. A modifier is a specification of what is written in any of the other sets of parentheses. For example, here it says Multi-part because the exercise this header belongs to has multiple questions. We want to keep the trend that the exercise type is general, while the modifier adds specificity. Your job is to follow these steps:\n\n1. Closely examine the header (specifically the exercise type and the modifier) and the question.\n2. Current exercise types may have types that are too specific or don't make sense. For example, \"Transcription and Meaning\" is not good and we would prefer Spelling Exercise. When you find an exercise type that is too specific or doesn't make sense: \nI. Replace it with a more general type (unless it's already there). Exercise types should always end with \"Exercise\".\nII. Take the specific type and try to write it/them in the modifier in the format With ... or Exercise that has ... or Exercise with ... or Exercise on ... or Activity on, Activity with...\nIII. When you add said modifier, make sure that you don't put it in pascal case\n3.  Carefully read the question to see if you can add a modifier that summarizes the question (using similar keywords to what is shown in (2. II.)).\n\n\nFor example:\n(French) (Write Country Names,  Transcription,  Activity,  Short Answer) D 6 (With Answer)\n*exercise about writing country names phonetically in a certain language*\n\nwould become:\n(French) (Writing Exercise, Phonetics Exercise, Activity, Short Answer Exercise) D 6 (Exercise on writing country names, With Answer)\n\nNOTE: You can replace/edit/remove exercise types but do not change modifiers that are already there, those are always correct.\n\n4.  Output the entire new header with the body of the question"

# -------------------------
# Parsers / Subparsers
# -------------------------
# Create the parser and subparsers
argparser = argparse.ArgumentParser(description="Extract text from images or reformat text using the OpenAI API.")
subparsers = argparser.add_subparsers(dest="key", help="Subcommand description")

# Subcommand for extracting text from images
parser_e = subparsers.add_parser("e", help="Extract text from images")
parser_e.add_argument("-sp", help="Split the given PDF into images before extracting text", action="store_true")
parser_e.add_argument('start_page', help="The starting page number", type=int)
parser_e.add_argument('end_page', help="The ending page number", type=int)

# Subcommand for improving extracted text
parser_et = subparsers.add_parser("et", help="Improve extracted text")

# Subcommand for annotating questions
parser_q = subparsers.add_parser("q", help="Annotate questions")
parser_q.add_argument('subject', help="The subject of the questions you want to annotate")

# Subcommand for submitting batch jobs
parser_sb = subparsers.add_parser("sb", help="Submit a specific batch job")
parser_sb.add_argument('batch_file_name', help="The name of the batch file (without the extension)")
parser_ab = subparsers.add_parser("ab", help="Submit all batch jobs")

# Subcommand for checking the status of a batch job
parser_s = subparsers.add_parser("s", help="Check the status of a batch job")
parser_s.add_argument('batch_job_id', help="The ID of the batch job")

# Subcommand for retrieving the results of a batch job
parser_r = subparsers.add_parser("r", help="Retrieve the results of a batch job")
parser_r.add_argument("-t", help="Retrieve the results from a text reformatting job", action="store_true")
parser_r.add_argument("-p", help="Retrieve the results from a text extraction job", action="store_true")

# Parse the arguments
args = argparser.parse_args()

if __name__ == "__main__":
    # Initialize the OpenAI API
    client = OpenAI(api_key="sk-proj-ltmkMm6qZ8oQCsusN5IOT3BlbkFJmsPopivPYwLtY7jlx5Pl")
    if args.key == "e":
        if args.sp:
            # Convert the PDFs to images
            PDF_to_images(PATH, args.start_page, args.end_page)

        # Upload the images to Cloudinary and get the URLs
        image_urls = [upload_image(path, CLOUDINARY_API, CLOUDINARY_SECRET) for path in glob.glob(f"{IMAGE_FOLDER}/*.png")]

        # Create multiple completions and store them in batches
        batch_items(BATCH_FOLDER, image_urls, PROMPT)
    elif args.key == "et":
        # Extract all the questions from the file
        questions = extract_raw_questions(HEADER, file=PATH)

        # Create multiple completions and store them in batches
        batch_items(BATCH_FOLDER, questions, PROMPT)
    elif args.key == "q":
        # Extract all the questions from the file
        with open(PATH, "r") as f:
            text = f.read()

        # Separate each question by their headers
        questions = re.split(r'\n(?=\(.*?\) \(.*?\))', text)
        # Remove empty strings and get the subject you want
        questions = [q.strip() for q in questions if q.strip() and args.subject in q]

        # Create multiple completions and store them in batches
        batch_items(BATCH_FOLDER, questions, PROMPT, args.subject, True)
    elif args.key == "sb":
        submit_batch(BATCH_FOLDER, client, args.batch_file_name)
    elif args.key == "ab":
        submit_batch(BATCH_FOLDER, client, files=True)
    elif args.key == 's':
        batch_job = client.batches.retrieve(args.batch_job_id)
        print(f"Batch job {args.batch_job_id} is {batch_job.status}")
    elif args.key == "r":  
        # Retrieve the results of the batch job
        with open("batch_job_id.txt", "r") as f:
            batch_job_ids = f.readlines()
        for batch_job_id in batch_job_ids:
            batch_job = client.batches.retrieve(batch_job_id.strip())
            result_file_id = batch_job.output_file_id
            result = client.files.content(result_file_id).content

            # Split the result into lines
            lines = result.splitlines()

            # Parse each line as a separate JSON object
            batch_results = [json.loads(line) for line in lines]
            # Iterate through each result and save the extracted text
            if args.t:
                with open("result.txt", "a") as f:
                    for result in batch_results:
                        task_id = result['custom_id']
                        subject, grade = task_id.split('_')
                        extracted_text = result['response']['body']['choices'][0]['message']['content']
                        f.write(extracted_text + '\n\n\n')
            elif args.p:
                with open("questions.txt", "a") as f:
                    for result in batch_results:
                        task_id = result['custom_id']
                        subject, grade, _ = task_id.split('_')
                        extracted_text = result['response']['body']['choices'][0]['message']['content']
                        exercises = extracted_text.split("*NEW*")[1:]
                        for exercise in exercises:
                            f.write(f"{subject} T D {grade} M\n")
                            f.write(exercise + '\n')
    else:
        argparser.print_help()