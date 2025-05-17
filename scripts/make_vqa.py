import os
import re
import csv
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
from dotenv import load_dotenv
import google.generativeai as genai

SRC_PATH        = '../data/csvs/curated.csv'
IMAGE_DEST_DIR  = '../data/curated_images'
DEST_PATH       = '../data/csvs/vqa.csv'

GENERATION_CONFIG = {
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 100,
    "max_output_tokens": 1000
}

QA_REGEX = re.compile(
    r"\s*Question\s*\d+:\s*(.*?)\s*Answer\s*\d+:\s*(.+)",
    re.IGNORECASE
)

def load_keys_and_model():
    load_dotenv()
    keys = os.getenv("GOOGLE_API_KEYS", "")
    API_KEYS = [k.strip() for k in keys.split(",") if k.strip()]
    if not API_KEYS:
        raise ValueError("No API keys found in .env or hardcoded.")
    
    key_cycle = cycle(API_KEYS)

    def configure_model(api_key):
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=GENERATION_CONFIG
        )

    model = configure_model(next(key_cycle))
    return API_KEYS, key_cycle, model

def prepare_prompt(image_info):
    name         = image_info.get('name', 'N/A')
    product_type = image_info.get('product_type', 'N/A')
    color        = image_info.get('color', 'N/A')
    keywords     = image_info.get('keywords', 'N/A')

    return f"""
        You are given an image, some metadata about that image, and a set of instructions. Follow the instructions exactly.
        Image: {image_info['img']}
        Metadata:
        Name: {name}
        Product Type: {product_type}
        Color: {color}
        Keywords: {keywords}
        Instructions:
        1. Generate 5 distinct questions.
        2. The questions can be answered by looking at the image or can be inferred by thinking.
        3. Difficult questions are preferred.
        4. Do not use quotation marks anywhere.
        5. The answer should exactly be 1 word.
        6. Provide the output strictly in the format given below.

        Question 1:
        Answer 1:
        Question 2:
        Answer 2:
        Question 3:
        Answer 3:
        Question 4:
        Answer 4:
        Question 5:
        Answer 5:
    """

def generate_qa(filename, df, model, api_key_cycle, API_KEYS):
    pairs = []
    img_path = Path(IMAGE_DEST_DIR) / filename

    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"Error: Image not found: {img_path}")
        return pairs
    except Exception as e:
        print(f"Error opening {img_path}: {e}")
        return pairs

    metadata = df.loc[filename].to_dict()
    metadata["img"] = img
    prompt = prepare_prompt(metadata)

    for _ in range(len(API_KEYS)):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            matches = QA_REGEX.findall(text)

            for q, a in matches:
                question = q.strip().rstrip('?.!')
                answer   = a.strip().lower()
                if question and answer:
                    pairs.append((question, answer))
            break

        except Exception as e:
            if "429" in str(e):  # rate limit
                next_key = next(api_key_cycle)
                model = genai.GenerativeModel("gemini-1.5-flash", generation_config=GENERATION_CONFIG)
                genai.configure(api_key=next_key)
                continue
            else:
                print(f"API error on {filename}: {e}")
                break

    return pairs

def initialize_output_csv():
    if not os.path.exists(DEST_PATH):
        with open(DEST_PATH, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['image_name','question','answer'])

    processed = set()
    try:
        existing = pd.read_csv(DEST_PATH)
        processed = set(existing['image_name'])
    except (pd.errors.EmptyDataError, FileNotFoundError):
        pass

    return processed

def write_qa_to_csv(filename, qa_pairs):
    with open(DEST_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for q, a in qa_pairs:
            writer.writerow([filename, q, a])

def main():
    df = pd.read_csv(SRC_PATH)
    df.set_index('image_name', inplace=True)

    API_KEYS, api_key_cycle, model = load_keys_and_model()
    processed = initialize_output_csv()

    all_files = list(df.index)
    to_process = [fn for fn in all_files if fn not in processed]

    for filename in tqdm(to_process, desc="VQA generation"):
        qa_pairs = generate_qa(filename, df, model, api_key_cycle, API_KEYS)
        if len(qa_pairs) == 5:
            write_qa_to_csv(filename, qa_pairs)
        else:
            print(f"Skipped {filename}: {len(qa_pairs)} Q&A pairs")

    out_df = pd.read_csv(DEST_PATH).sort_values(['image_name', 'question']).reset_index(drop=True)
    out_df.to_csv(DEST_PATH, index=False)
    print(f"\nDone. Total Q&A rows: {len(out_df)}")

if __name__ == "__main__":
    main()