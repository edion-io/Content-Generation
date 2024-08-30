# Copyright (C) 2024  Edion Management Systems
import re
import spacy
import glob
import fitz
import json
import cloudinary
import cloudinary.uploader
import cloudinary.api
from openai import OpenAI
from PIL import Image
import io
import tiktoken
from copy import deepcopy

SUBJECTS = ["Computer Science", "Science", "Mathematics", "Social Studies", "History", "Geography",
            "Spanish", "French", "German", "Dutch", "English"]

def find_first_number(string: str) -> str:
    """Find the first occurrence of a number in a string.

    Args:
        string (str): The string to search.

    Returns:
        str: The first number found in the string, or None if no number is found.
    """
    # Compile a regular expression pattern to match numbers
    pattern = re.compile(r'\d+')
    # Search for the first occurrence of the pattern in the string
    match = pattern.search(string)
    # If a match is found, return it; otherwise, return None
    if match:
        return match.group()
    return None

def has_imperative(sentence: str) -> bool:
    """Check if the sentence contains an imperative verb.

    Args:
        sentence (str): The sentence to check.

    Returns:
        bool: True if the sentence contains an imperative verb, False otherwise.
    """
    # Load the English NLP model
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    # Iterate over all tokens to find imperative verbs
    for token in doc:
        # Check if the token is a verb in base form and is the main verb (ROOT)
        if token.tag_ == 'VB' and token.dep_ == 'ROOT':
            # Check if there is no explicit subject for this verb
            if not any(child.dep_ == 'nsubj' for child in token.children):
                return True
    return False

def within_bounds(value: float, lower_bound: float, upper_bound: float, uncertainty: int) -> bool:
    """ Check if a value is within a specified range with a given uncertainty.

    Args:
        value (float): The value to check.
        lower_bound (float): The lower bound of the range.
        upper_bound (float): The upper bound of the range.
        uncertainty (int): The uncertainty or tolerance for the value.

    Returns:
        bool: True if the value is within the specified range, False otherwise.
    """
    return lower_bound - uncertainty <= value <= upper_bound + uncertainty

def within_distance(value: float, edge: float, threshold: int) -> bool:
    """ Check if a value is within a specified distance from an edge.

    Args:
        value (float): The value to check.
        edge (float): The edge value.
        threshold (int): The distance threshold.

    Returns:
        bool: True if the value is within the specified distance from the edge, False otherwise.
    """
    return abs(value - edge) <= threshold

def within_block_distance(edge1: float, edge2: float, value1: float, value2: float, lower: float, upper: float, threshold: int) -> bool:
    """ Check if two blocks are close to each other within a specified distance.

    Args:
        edge1 (float): The edge of the first block.
        edge2 (float): The edge of the second block.
        value1 (float): The value of the first block.
        value2 (float): The value of the second block.
        lower (float): The lower bound of the range.
        upper (float): The upper bound of the range.
        threshold (int): The distance threshold.
    
    Returns:
        bool: True if the blocks are close to each other within the specified distance, False otherwise
    """
    return within_distance(edge1, edge2, threshold) \
        and within_bounds(value1, lower, upper, threshold) \
        and within_bounds(value2, lower, upper, threshold)

def are_blocks_close(block1: dict, block2: dict, threshold=15) -> bool:
    """ Check if two blocks are close to each other within a specified distance,
        regardless of their relative positions.

    Args:
        block1 (dict): The first block.
        block2 (dict): The second block.
        threshold (int): The distance threshold.
    
    Returns:
        bool: True if the blocks are close to each other within the specified distance, False otherwise
    """
    # Identify where block 2 is relative to block 1
    if block2['bbox'][1] >= block1['bbox'][3]:
        # Block 2 is below block 1
        return within_block_distance(block2['bbox'][1], block1['bbox'][3], block2['bbox'][0], \
                            block2['bbox'][2], block1['bbox'][0], block1['bbox'][2], threshold)
    if block2['bbox'][3] <= block1['bbox'][1]:
        # Block 2 is above block 1
        return within_block_distance(block2['bbox'][3], block1['bbox'][1], block2['bbox'][0], \
                            block2['bbox'][2], block1['bbox'][0], block1['bbox'][2], threshold)
    if block2['bbox'][2] <= block1['bbox'][0]:
        # Block 2 is to the left of block 1
        return within_block_distance(block2['bbox'][2], block1['bbox'][0], block2['bbox'][1], \
                            block2['bbox'][3], block1['bbox'][1], block1['bbox'][3], threshold)
    if block2['bbox'][0] >= block1['bbox'][2]:
        # Block 2 is to the right of block 1
        return within_block_distance(block2['bbox'][0], block1['bbox'][2], block2['bbox'][1], \
                            block2['bbox'][3], block1['bbox'][1], block1['bbox'][3], threshold)
    return True

def combine_blocks(blocks: list) -> dict:
    """ Combine multiple blocks into a single block by concatenating the text
        and adjusting the bounding box.
    
    Args:
        blocks (list): A list of blocks to combine.

    Returns:
        dict: A combined block with the concatenated text and adjusted bounding box.
    """    
    combined_text = "\n".join(block['text'] for block in blocks)
    combined_bbox = [
        min(block['bbox'][0] for block in blocks),
        min(block['bbox'][1] for block in blocks),
        max(block['bbox'][2] for block in blocks),
        max(block['bbox'][3] for block in blocks)
    ]
    return {'text': combined_text, 'bbox': combined_bbox}

def is_near_top_or_bottom(block: dict, page_height: float, threshold=200) -> tuple:
    """ Check if a block is near the top or bottom of the page within a specified threshold.

    Args:
        block (dict): The block to check.
        page_height (float): The height of the page.
        threshold (int): The distance threshold.
    
    Returns:
        tuple: A tuple of booleans indicating if the block is near the top and bottom of the page.
    """
    top_distance = block['bbox'][1]
    bottom_distance = page_height - block['bbox'][3]
    near_top = top_distance <= threshold
    near_bottom = bottom_distance <= threshold
    return near_top, near_bottom

def PDF_to_images(folder: str, min: int, max: int) -> None:
    """Convert PDF files in a folder to images.

    Args:
        folder (str): The folder containing the PDF files.
        min (int): The minimum page number to process.
        max (int): The maximum page number to process.
    """
    # Split the PDFs into images
    for i, path in enumerate(glob.glob(f"{folder}/Singapore Primary*.pdf")):
        grade = find_first_number(path)
        # Save the subject
        subject = folder.split('/')[1]
        doc = fitz.open(path)
        for page_num in range(doc.page_count):
            # Skip the first few pages
            if page_num < min or page_num > max:
                continue
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            pix.save(f'imgs/{subject}_{grade}_{(10 ** i + 1) * page_num + 1}.png')

def upload_image(image_path: str, api_key: str, api_secret: str) -> tuple:
    """ Upload an image to Imgur and get the URL.

    Args:
        image_path (str): The path to the image file.
        api_key (str): The Cloudinary API key.
        api_secret (str): The Cloudinary API secret.

    Returns:
        tuple: A tuple containing the subject, grade, and the URL of the uploaded image.
    """
    # Configure Cloudinary with your credentials
    cloudinary.config(
      cloud_name='duxnyytva',
      api_key=api_key,
      api_secret=api_secret 
    )

    name = image_path.split('/')[-1]
    subject, grade = name.split('_')[:2]

    try:
        response = cloudinary.uploader.upload(image_path, public_id=name)
        print("Upload Successful:", response['url'])
        return subject, grade, response['url']
    except Exception as e:
        print(f"Error uploading file {image_path}: {e}")
        return None

def extract_raw_questions(header: str, file: str) -> list:
    """ Extracts all the questions from a file or text.

    Args:
        header (str): The header to search for.
        file (str): The file to extract questions from.

    Returns:
        list: A list of questions.
    """
    first = True
    segment = ''
    prompts = []
    with open (file, "r") as f:
        for line in f:
            if header in line:
                if not first:
                    prompts.append(segment)
                else:
                    first = False
                segment = line
            else:
                segment += line
    return prompts
    
def batch_items(batch_folder: str, items: list, prompt: str, placeholder = 'text', is_text = False) -> None:
    """ Create a batch of chat completion tasks for a list of texts or images.

    Args:
        batch_folder (str): The folder to save the batch files to.
        items (list): A list of texts or image urls.
        prompt (str): The prompt for the chat completion task.
        placeholder (str): The placeholder title for the batch (generally a subject or just 'text').
        is_text (bool): A flag indicating if the items are text or images.
    """
    batches, current_batch, current_tokens = [], [], 0
    if is_text:
        # Process each text item
        for i, text in enumerate(items):
            current_tokens = batch(text, f"{placeholder}_{i}", current_tokens, current_batch, batches, prompt, is_text)
    else:
        # Process each image item
        for i, (subject, grade, url) in enumerate(items):
            current_tokens = batch(url, f"{subject}_{grade}_{i}", current_tokens, current_batch, batches, prompt)

    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)

    # Save the smaller batches as separate files
    for i, items in enumerate(batches):
        with open(f'{batch_folder}/batch_{i+1}.jsonl', 'w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')

def batch(content: str, name: str, current_tokens: int, current_batch: list, batches: list, prompt: str, is_text = False) -> int:
    """ Create a single chat completion task for a text or image.
    
    Args:
        content (str): The text or image URL to process.
        name (str): The name of the task.
        current_tokens (int): The current number of tokens in the batch.
        current_batch (list): The current batch of tasks.
        batches (list): The list of batches.
        prompt (str): The prompt for the chat completion task.
        is_text (bool): A flag indicating if the content is text or an image.

    Returns:
        int: The updated number of tokens in the batch.
    """
    # Build the messages for the chat completion task
    messages = [{
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": content
                } if is_text else {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": content
                            }
                        }
                    ]
                }]
    
    # Create one task for the batch
    task = ({
        "custom_id": name,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # Chat Completions API call
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 3000,
            "messages": messages
        }
    })  

    # Get the encoding for the model
    encoding = tiktoken.get_encoding("cl100k_base")
    # Encode the text to get the tokens
    task_tokens = len(encoding.encode(prompt)) + (len(encoding.encode(content)) if is_text else 0)
    # Check if the current batch is full
    if current_tokens + task_tokens > 190000:
        # If it is full, add the current batch to the list of batches
        batches.append(deepcopy(current_batch))
        current_batch.clear()
        current_batch.append(task)
        current_tokens = task_tokens
    else:
        # Otherwise, add the task to the current batch
        current_batch.append(task)
        current_tokens += task_tokens
    
    return current_tokens

def submit_batch(batch_folder: str, client: OpenAI, file=None, files=False) -> None:
    """Submit a batch job to process a batch of chat completion tasks.

    Args:
        batch_folder (str): The folder containing the batch files.
        client (OpenAI): The OpenAI client.
        file (str): The name of the batch file to submit.
        files (bool): A flag indicating if multiple batch files should be submitted.
    """
    if files:
        # Submit all batch files
        for file in glob.glob(f'{batch_folder}/batch_*.jsonl'):
            submit(client, file)
    else:
        # Submit a single batch file
        submit(client, f"{batch_folder}/{file}.jsonl")

def submit(client: OpenAI, file: str) -> None:
    """Submit a batch job to process a batch of chat completion tasks.

    Args:
        client (OpenAI): The OpenAI client.
        file (str): The name of the batch file to submit.
    """

    # Create a batch job for the file
    batch_file = client.files.create(
        file=open(file, "rb"),
        purpose="batch"
    )
    # Create the job to process the batch
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    # Check if the file is empty or not
    try:
        with open("batch_job_id.txt", "r") as f:
            first_char = f.read(1)  # Read the first character
    except FileNotFoundError:
        # If the file doesn't exist, treat it as empty
        first_char = None

    # Now, open the file in append mode to write the job ID
    with open("batch_job_id.txt", "a") as f:
        # If the file is empty or doesn't exist, write the id without a newline
        if not first_char:
            f.write(batch_job.id)
        else:
            f.write(f"\n{batch_job.id}")

def modify_jsonl(file_path: str, output_path: str, new_prompt: str):
    """ Modify the content of a JSONL file.

    Args:
        file_path (str): The path to the input JSONL file.
        output_path (str): The path to the output JSONL file.
        new_prompt (str): The new prompt to replace the content with.
    """
    with open(file_path, 'r') as f, open(output_path, 'w') as out:
        for line in f:
            obj = json.loads(line)
            obj['body']['messages'][0]['content'] = new_prompt
            json.dump(obj, out)
            out.write('\n')

def combine(images: list, output_path: str) -> None:
    """ Combines images into a single image.

    Args:
        images (list): The list of images to combine.
        output_path (str): The path to save the combined image to.
    """
    # Assuming all images are the same width and height
    width, height = images[0].size

    # Create a new image with appropriate height to hold all the images
    total_height = height * len(images)
    combined_image = Image.new('RGB', (width, total_height))

    # Paste each image below the previous one
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += height

    # Save the combined image
    combined_image.save(output_path)

def get_images(file: str, pages: list, offset: int, spec: str) -> list:
    """ Extracts related images (in a range) from a PDF file and combines them into a single image.

    Args:
        file (str): The path to the PDF file.
        pages (list): The list of page ranges to extract images from.
        offset (int): The offset to add to the page number.
        spec (str): A string to add to the image name to differentiate it.

    Returns:
        list: The list of images extracted from the PDF.
    """
    images = []
    i = 0
    doc = fitz.open(file)
    start, stop = pages.pop(0)
    for page_num in range(doc.page_count):
        # Check if the page number is within the range
        if start + offset <= page_num <= stop + offset:
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("ppm") # Get pixmap in PPM format
            img = Image.open(io.BytesIO(img_data))  # Convert bytes to PIL Image
            images.append(img)

            # Combine images if the range is complete
            if page_num == stop + offset:
                combine(images, f'imgs/{spec}_{i}.png')
                i += 1
                images.clear()
                if len(pages) == 0:
                    break
                start, stop = pages.pop(0)
        else:
            continue
    return images