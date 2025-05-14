import os
import csv
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# --- Config ---
INPUT_PATH        = '../data/curated.csv'
TARGET_IMAGE_DIR  = '../data/curated_images'
OUTPUT_PATH       = '../data/vqa.csv'

# Load curated metadata
df = pd.read_csv(INPUT_PATH)
df.set_index('filename', inplace=True)

# Load API keys
load_dotenv()
api_keys_str = os.getenv("GOOGLE_API_KEYS", "")
API_KEYS = [k.strip() for k in api_keys_str.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("\nNo API keys found.")
api_key_cycle = cycle(API_KEYS)

# Gemini / Gemini‐1.5‐flash config
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 5000
}

def configure_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )

# initialize model
current_api_key = next(api_key_cycle)
model = configure_model(current_api_key)

# Regex for parsing outputs
QA_REGEX = re.compile(
    r"\d+\.\s*Question\s*\d+:\s*(.*?)\s*Answer\s*\d+:\s*(.+)",
    re.IGNORECASE
)

def generate_qa(filename):
    global model, current_api_key

    pairs = []
    img_path = Path(TARGET_IMAGE_DIR) / filename

    # load image
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"\nError: Image not found: {img_path}")
        return pairs
    except Exception as e:
        print(f"\nError opening {img_path}: {e}")
        return pairs

    # fetch metadata
    row = df.loc[filename]
    name         = row.get('name', 'N/A')
    product_type = row.get('product_type', 'N/A')
    color        = row.get('color', 'N/A')
    keywords     = row.get('keywords', 'N/A')

    # build prompt
    prompt = [
        "You are given an image, some metadata about that image, and a set of instructions. Follow the instructions exactly.",
        "\nImage:", img,
        "\nMetadata:",
        f"- Name: {name}",
        f"- Product Type: {product_type}",
        f"- Main Color Provided: {color}",
        f"- Keywords: {keywords}\n",
        "\nInstructions:",
        "1. Generate exactly 5 DISTINCT questions.",
        "2. Questions must be answerable by the image or metadata.",
        "3. Do NOT add quotation marks.",
        "4. Each answer must be a single word.",
        "5. Provide the output STRICTLY in the following format, with each question and answer pair clearly marked. Do not include any other text before or after this numbered list:\n",
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
        """
    ]

    # try once per key until success or exhaustion
    for _ in range(len(API_KEYS)):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            # parse out Q/A
            matches = QA_REGEX.findall(text)
            for q, a in matches:
                question = q.strip().rstrip('?.!')
                answer   = a.strip().lower()
                if question and answer:
                    pairs.append((question, answer))

            if len(pairs) < 5:
                print(f"\nWarning: only {len(pairs)} pairs parsed for {filename}")
            break

        except Exception as e:
            if "429" in str(e):
                # rate limited → rotate key
                current_api_key = next(api_key_cycle)
                model = configure_model(current_api_key)
                continue
            else:
                print(f"\nAPI error on {filename}: {e}")
                break

    return pairs

def main():
    # initialize or resume CSV
    processed = set()
    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['filename','question','answer'])
    else:
        try:
            chk = pd.read_csv(OUTPUT_PATH)
            processed = set(chk['filename'])
        except pd.errors.EmptyDataError:
            pass

    all_files = list(df.index)
    to_process = [fn for fn in all_files if fn not in processed]

    for filename in tqdm(to_process, desc="VQA generation"):
        qa = generate_qa(filename)
        if len(qa) == 5:
            with open(OUTPUT_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for q,a in qa:
                    writer.writerow([filename, q, a])
        else:
            print(f"\nSkipped {filename}: {len(qa)} Q&A pairs")

    # final sort & rewrite
    out_df = pd.read_csv(OUTPUT_PATH).sort_values(['filename','question']).reset_index(drop=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. Total Q&A rows: {len(out_df)}")

if __name__ == "__main__":
    main()