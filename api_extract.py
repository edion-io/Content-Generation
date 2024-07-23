# Copyright (C) 2024  Edion Management Systems
from openai import OpenAI
from utils import PDF_to_images, upload_image, batch_vision
import imgurpython
import glob
import sys
import json

# Specify the page range to process
PAGE_MIN = 5
PAGE_MAX = 99
# Specify the folder containing the PDF files
FOLDER = 'books/Computer Science/Primary'
# Specify the folder containing the images
IMAGE_FOLDER = 'imgs'
# Specify the Imgur client ID and secret
IMGUR_CLIENT_ID = "ad11d8fb10a845b"
IMGUR_SECRET = "9ee935061329341c51dbc1e7e7f6b5b20d1f6503"
# Specify the prompt for the chat completion task
prompt = "In the given image, extract any text related to 'Test', 'Activity', 'Think Again', 'Extra Challenge' or 'Explore More and output the occurence(s) in this format:\n *NEW*\n[Replace with Activity, Think Again, Extra Challenge or Explore More] \n [Replace with Extracted Text]\n If the extracted text relies on an image, give a description of the image in the format '[STARTDGM] [Replace with description of diagram] [STOPDGM]' and add it below the extract text. If there are multiple occurrences, simply separate them with a newline. If the extracted text relies on surrounding text (e.g an Activity uses text laid out on the page in the form of an image or graphic) then provide the text/sentences as well. If there are no occurrences of any of the four targets on the image, then output 'No text'."

if __name__ == "__main__":
    # Initialize the OpenAI API
    client = OpenAI("sk-proj-ltmkMm6qZ8oQCsusN5IOT3BlbkFJmsPopivPYwLtY7jlx5Pl")

    if sys.argv[1] == "-e":
        if len(sys.argv) != 4:
            print("Usage: python api_extract.py -e <min_page> <max_page>")
            sys.exit(1)

        # Convert the PDFs to images
        PDF_to_images(FOLDER, sys.argv[2], sys.argv[3])

        # Upload the images to Imgur and get the URLs
        client = imgurpython.ImgurClient(IMGUR_CLIENT_ID, IMGUR_SECRET)
        image_urls = [upload_image(client, path) for path in glob.glob(f"{IMAGE_FOLDER}/*.png")]

        # Create a chat completion task for each image and batch it
        batch_file = client.files.create(
          file=open(batch_vision(image_urls, prompt), "rb"),
          purpose="batch"
        )

        # Create the job to process the batch
        batch_job = client.batches.create(
          input_file_id=batch_file.id,
          endpoint="/v1/chat/completions",
          completion_window="24h"
        )

        # Store the id of the batch job
        with open("batch_job_id.txt", "w") as f:
            f.write(batch_job.id)
    elif sys.argv[1] == "-r":
        # Retrieve the results of the batch job
        with open("batch_job_id.txt", "r") as f:
            batch_job_id = f.read()
        batch_job = client.batches.retrieve(batch_job_id)
        result_file_id = batch_job.output_file_id
        result = client.files.content(result_file_id).content

        # Parse the JSON content
        batch_results = json.loads(result)

        # Iterate through each result and save the extracted text
        with open("questions.txt", "a") as f:
            for result in batch_results['results']:
                task_id = result['custom_id']
                subject, grade, _ = task_id.split('_')
                extracted_text = result['output']['text']
                f.write(f"{subject} T D {grade} M\n")
                f.write(extracted_text)
    else:
        print("Invalid argument. Use -e to extract text or -r to retrieve results.")