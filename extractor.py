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
PATH = 'questions.txt'
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
PROMPT = "1. Extract all exercises from the pages .\n2. Output them in this format:\n*NEW*\n[Max 5 words describing the type of exercise]\n[Replace with extracted text]\n[Situational diagram description (see instruction #4 below)]\n\n 3. Some exercises continue over to another page, make sure you get that too (should not be separate). 4. If an activity or exercise needs or refers to one or more images or tables, add a description of the image(s) to the above template in the form:\n[STRDGRM] [Detailed description of the image that a blind person can use to visualize what is needed, without even seeing the image] [STPDGRM]\n\n5. Only output what is asked of you."

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
        questions = re.split(r'(?m)^(' + "|".join(re.escape(s) for s in SUBJECTS) + ')', text)
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
        print(batch_job)
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
                with open("qs.txt", "a") as f:
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