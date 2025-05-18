import torch
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering

MODEL_POINT = "pratster/salesforce_blip_fine_tuned"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)
    # df = df.sample(n=1000, random_state=7).reset_index(drop=True) # should be commented out

    # Load processor and question-answering model, move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(MODEL_POINT, use_fast=True)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_POINT).to(device)
    model.eval()

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            # Prepare inputs for generative QA
            inputs = processor(image, question, return_tensors="pt", truncation=True).to(device)
            # Generate answers using the QA model's generation head
            output_ids = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
            # Decode to string and basic post-processing
            answer = processor.decode(output_ids[0], skip_special_tokens=True)
        except Exception:
            answer = "error"
        # Ensure answer is one word and in English
        answer = answer.split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
