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
    for path in glob.glob(f"{folder}/Oxford*.pdf"):
        grade = find_first_number(path)
        # Save the subject
        subject = folder.split('/')[1]
        doc = fitz.open(path)
        for page_num in range(doc.page_count):
            # Skip the first few pages (e.g., table of contents, glossary)
            if page_num < min or page_num > max:
                continue
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            pix.save(f'imgs/{subject}_{grade}_page_{page_num + 1}.png')

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

def batch_vision(urls: list, prompt: str) -> str:
    """Create a batch of chat completion tasks for a list of image URLs.
    
    Args:
        urls (list): A list of image URLs.
        prompt (str): The prompt for the chat completion task.
        filename (str): The name of the file to save the batch file to.

    Returns:
        str: The name of the file containing the batch of tasks.
    """
    batches, current_batch = [], []
    current_tokens = 0
    item_tokens = len(prompt.split())
    for i, (subject, grade, url) in enumerate(urls):
        item = ({
        "custom_id": f"{subject}_{grade}_{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # Chat Completions API call
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 3000,
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url
                            }
                        },
                    ],
                }
            ]            
        }
    })
        if current_tokens + item_tokens > 190000:
            batches.append(current_batch)
            current_batch = [item]
            current_tokens = item_tokens
        else:
            current_batch.append(item)
            current_tokens += item_tokens

    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)

    # Save the smaller batches as separate files
    for i, batch in enumerate(batches):
        with open(f'tasks/batch_{i+1}.jsonl', 'w') as f:
            for item in batch:
                f.write(json.dumps(item) + '\n')

def submit_batch(client: OpenAI, file=None, files=False) -> None:
    """Submit a batch job to process a batch of chat completion tasks.

    Args:
        client (OpenAI): The OpenAI client.
        file (str): The name of the batch file to submit.
        files (bool): A flag indicating if multiple batch files should be submitted.
    """
    if files:
        # Submit all batch files
        for file in glob.glob('tasks/input_file_batch_*.json'):
            submit(client, file)
    else:
        # Submit a single batch file
        submit(client, f"tasks/{file}.jsonl")

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
    # Store the id of the batch job
    with open("batch_job_id.txt", "a") as f:
        f.write(batch_job.id+"\n")

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