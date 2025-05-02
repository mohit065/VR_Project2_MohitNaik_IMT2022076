import re
import os
import csv
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from itertools import cycle
from dotenv import load_dotenv
import google.generativeai as genai

IMAGE_DIR = '../data/images/small'
INPUT_PATH = '../data/curated.csv'
OUTPUT_PATH = '../data/vqa_data_flat.csv'

df = pd.read_csv(INPUT_PATH)
df.set_index('path', inplace=True)

load_dotenv()
api_keys_str = os.getenv("GOOGLE_API_KEYS", "")
API_KEYS = [key.strip() for key in api_keys_str.split(",") if key.strip()]
if not API_KEYS:
    raise ValueError("No API keys found in GOOGLE_API_KEYS environment variable.")
api_key_cycle = cycle(API_KEYS)
current_api_key = next(api_key_cycle)

generation_config = {"temperature": 0.4, "top_p": 1, "top_k": 32, "max_output_tokens": 2000}
def configure_model_with_key(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )

model = configure_model_with_key(current_api_key)

def generate_qa(img_path):
    global model, current_api_key
    generated_pairs = []
    full_path = str(Path(IMAGE_DIR) / img_path).replace('\\', '/')

    try:
        img = Image.open(full_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {full_path}")
        return generated_pairs
    except Exception as e:
        print(f"Error opening image {full_path}: {e}")
        return generated_pairs

    try:
        row = df.loc[img_path]
    except KeyError:
        print(f"Warning: No metadata found in DataFrame for {img_path}")
        return generated_pairs

    name = row.get('name', 'N/A')
    product_type = row.get('product_type', 'N/A')
    color = row.get('color', 'N/A')
    keywords = row.get('keywords', 'N/A')

    prompt_parts = [
        "Given the image and the metadata below, generate 5 distinct visual questions.",
        "Metadata:",
        f"- Name: {name}",
        f"- Product Type: {product_type}",
        f"- Main Color Provided: {color}",
        f"- Keywords: {keywords}\n",
        "Instructions:",
        "1. Analyze the provided image and the metadata.",
        "2. Generate exactly 5 distinct questions about prominent visual features, objects, colors, materials, or attributes clearly visible in the image.",
        "3. Each question MUST have a single-word answer directly verifiable from the image.",
        "4. The 5 questions generated MUST be different from each other, and make sure that the questions are answerable just by looking at the image.",
        "5. Provide the output strictly in the following numbered format, with each question and answer pair clearly marked. Do not include any other text before or after this numbered list:\n",
        """
        1.
        Question 1: [Your first question here]
        Answer 1: [Your single-word answer here]
        2.
        Question 2: [Your second question here]
        Answer 2: [Your single-word answer here]
        3.
        Question 3: [Your third question here]
        Answer 3: [Your single-word answer here]
        4.
        Question 4: [Your fourth question here]
        Answer 4: [Your single-word answer here]
        5.
        Question 5: [Your fifth question here]
        Answer 5: [Your single-word answer here]
        """,
        "\nImage:",
        img,
    ]

    max_attempts = len(API_KEYS)
    for _ in range(max_attempts):
        try:
            response = model.generate_content(prompt_parts)
            response_text = response.text.strip()

            pattern = re.compile(
                r"\d+\.\s*Question\s*\d+: (.*?)\s*Answer\s*\d+: (\w+)",
                re.IGNORECASE
            )
            matches = pattern.findall(response_text)

            if matches:
                for q, a in matches:
                    question = q.strip().rstrip('?.!')
                    answer = a.strip().lower()
                    if question and answer:
                        generated_pairs.append((question, answer))
                if len(generated_pairs) < 5:
                    print(f"Warning: Parsed fewer than 5 Q&A pairs ({len(generated_pairs)}).")
            else:
                print(f"Warning: Could not parse Q&A pairs using regex.")
            break

        except Exception as e:
            if "429" in str(e):
                current_api_key = next(api_key_cycle)
                model = configure_model_with_key(current_api_key)
                continue
            else:
                print(f"Error during Gemini API call or processing for {img_path}: {e}")
                break

    return generated_pairs

def looper():
    processed_paths = set()

    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'question', 'answer'])
        print(f"Initialized new checkpoint file at {OUTPUT_PATH}.")
    else:
        try:
            checkpoint_df = pd.read_csv(OUTPUT_PATH)
            processed_paths = set(checkpoint_df['image_path'].unique())
            print(f"Resuming from checkpoint: {len(processed_paths)} images already processed.")
        except pd.errors.EmptyDataError:
            print("Warning: Output file exists but is empty. Initializing with header.")
            with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'question', 'answer'])

    remaining_paths = [p for p in df.index if p not in processed_paths]

    for img_path in tqdm(remaining_paths, desc="Processing images", unit="image"):
        qa_pairs = generate_qa(img_path)
        if len(qa_pairs) == 5:
            with open(OUTPUT_PATH, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for q, a in qa_pairs:
                    writer.writerow([img_path, q, a])
        else:
            print(f"Skipped {img_path}: fewer than 5 Q&A pairs.")

def main():
    looper()
    df_flat = pd.read_csv(OUTPUT_PATH)
    df_flat = df_flat.sort_values(by='image_path').reset_index(drop=True)
    df_flat.to_csv(OUTPUT_PATH, index=False)
    print(f"Finished VQA generation. Total rows: {len(df_flat)} (5 per image).")

if __name__ == "__main__":
    main()