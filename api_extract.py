# Copyright (C) 2024  Edion Management Systems
from openai import OpenAI
from utils import PDF_to_images, upload_image, batch_vision, submit_batch
import glob
import sys
import json

# Specify the folder containing the PDF files
FOLDER = 'books/English/Primary'
# Specify the folder containing the images
IMAGE_FOLDER = 'imgs'
# Specify the Imgur client ID and secret
CLOUDINARY_API = "667797151493891"
CLOUDINARY_SECRET = "WuKdiXBzcwzUgOsdOey5J9E8k7c"
# Specify the prompt for the chat completion task
prompt = "Extract any exercise you see on the page. Output the occurrence(s) in this format:\n *NEW*\n[Replace with type of exercise (be detailed)] \n [Replace with question]\n If the extracted text needs an image to make sense, give a description of the image in the format '[STARTDGM] [Replace with description of diagram] [STOPDGM]' below the extracted text. Separate multiple occurrences with a newline. If there are no occurrences then output 'No text\n'."

if __name__ == "__main__":
    # Initialize the OpenAI API
    client = OpenAI(api_key="sk-proj-ltmkMm6qZ8oQCsusN5IOT3BlbkFJmsPopivPYwLtY7jlx5Pl")

    if sys.argv[1] == "-e":
        if len(sys.argv) != 4:
            print("Usage: python api_extract.py -e <min_page> <max_page>")
            sys.exit(1)

        # Convert the PDFs to images
        PDF_to_images(FOLDER, int(sys.argv[2]), int(sys.argv[3]))

        # Upload the images to Cloudinary and get the URLs
        image_urls = [upload_image(path, CLOUDINARY_API, CLOUDINARY_SECRET) for path in glob.glob(f"{IMAGE_FOLDER}/*.png")]

        # Create multiple completions and store them in batches
        batch_vision(image_urls, prompt)
    elif sys.argv[1] == "-sb":
        if len(sys.argv) != 3:
            print("Usage: python api_extract.py -b <batch_file_name>")
            sys.exit(1)
        submit_batch(client, sys.argv[2])
    elif sys.argv[1] == "-ab":
        submit_batch(client, files=True)
    elif sys.argv[1] == '-s':
        if len(sys.argv) != 3:
            print("Usage: python api_extract.py -s <batch_job_id>")
            sys.exit(1)
        batch_job = client.batches.retrieve(sys.argv[2])
        print(batch_job)
    elif sys.argv[1] == "-r":
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
        print("Invalid argument. Use -e to extract text, -r to retrieve results,\n -sb to submit a batch job or -ab to submit all batches")