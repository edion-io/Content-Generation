# Copyright (C) 2024  Edion Management Systems
from openai import OpenAI
from utils import upload_image, batch_items, submit_batch, extract_raw_questions, PDF_to_images
import glob
import sys
import json

# Specify the folder containing any relevant files
FOLDER = 'books/Math/Primary'
# Specify the folder containing the images
IMAGE_FOLDER = 'imgs'
# Specify the batch folder
BATCH_FOLDER = 'tasks'
# Specify the question header for raw questions
HEADER = "German T D G (With Answer)"
# Specify the Cloudinary client ID and secret
CLOUDINARY_API = "667797151493891"
CLOUDINARY_SECRET = "WuKdiXBzcwzUgOsdOey5J9E8k7c"

# Specify the prompt for the chat completion task
PROMPT = "1. Extract all exercises from the pages .\n2. Output them in this format:\n*NEW*\n[Max 5 words describing the type of exercise]\n[Replace with extracted text]\n[Situational diagram description (see instruction #4 below)]\n\n 3. Some exercises continue over to another page, make sure you get that too (should not be separate). 4. If an activity or exercise needs or refers to one or more images or tables, add a description of the image(s) to the above template in the form:\n[STRDGRM] [Detailed description of the image that a blind person can use to visualize what is needed, without even seeing the image] [STPDGRM]\n\n5. Only output what is asked of you."

if __name__ == "__main__":
    # Initialize the OpenAI API
    client = OpenAI(api_key="sk-proj-ltmkMm6qZ8oQCsusN5IOT3BlbkFJmsPopivPYwLtY7jlx5Pl")
    if sys.argv[1] == "-e":
        if len(sys.argv) < 4:
            print("Usage: python api_extract.py -e [-s] <min_page> <max_page>")
            sys.exit(1)

        if sys.argv[2] != "-s":
            # Convert the PDFs to images
            PDF_to_images(FOLDER, int(sys.argv[2]), int(sys.argv[3]))

        # Upload the images to Cloudinary and get the URLs
        image_urls = [upload_image(path, CLOUDINARY_API, CLOUDINARY_SECRET) for path in glob.glob(f"{IMAGE_FOLDER}/*.png")]

        # Create multiple completions and store them in batches
        batch_items(BATCH_FOLDER, image_urls, PROMPT)
    elif sys.argv[1] == "-et":
        if len(sys.argv) != 2:
            print("Usage: python api_extract.py -et")
            sys.exit(1)
        # Extract all the questions from the file
        questions = extract_raw_questions(HEADER, file=FOLDER)

        # Create multiple completions and store them in batches
        batch_items(BATCH_FOLDER, questions, PROMPT)
    elif sys.argv[1] == "-sb":
        if len(sys.argv) != 3:
            print("Usage: python api_extract.py -b <batch_file_name>")
            sys.exit(1)
        submit_batch(BATCH_FOLDER, client, sys.argv[2])
    elif sys.argv[1] == "-ab":
        submit_batch(BATCH_FOLDER, client, files=True)
    elif sys.argv[1] == '-s':
        if len(sys.argv) != 3:
            print("Usage: python api_extract.py -s <batch_job_id>")
            sys.exit(1)
        batch_job = client.batches.retrieve(sys.argv[2])
        print(batch_job)
    elif sys.argv[1] == "-r":  
        if len(sys.argv) != 2:
            print("Usage: python api_extract.py -r [-t | -p]")
            sys.exit(1)
        # Retrieve the results of the batch job
        with open("batch_job_id.txt", "r") as f:
            batch_job_ids = f.readlines()
        for batch_job_id in batch_job_ids:
            batch_job = client.batches.retrieve(batch_job_id)
            result_file_id = batch_job.output_file_id
            result = client.files.content(result_file_id).content

            # Split the result into lines
            lines = result.splitlines()

            # Parse each line as a separate JSON object
            batch_results = [json.loads(line) for line in lines]
        # Iterate through each result and save the extracted text
        if sys.argv[2] == "-t":
            with open("questions.txt", "a") as f:
                for result in batch_results:
                    task_id = result['custom_id']
                    subject, grade = task_id.split('_')
                    extracted_text = result['response']['body']['choices'][0]['message']['content']
                    f.write(extracted_text + '\n\n')
        elif sys.argv[2] == "-p":
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
        print("Invalid argument. Use -e to extract text, -et to improve text,\n -r to retrieve results, -sb to submit a batch job, -ab to submit all batches,\n or -s to check the status of a batch job.")