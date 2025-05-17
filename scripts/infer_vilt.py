import os
import csv
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import ViltProcessor, ViltForQuestionAnswering

MODEL_DIR     = "../data/vilt_ft"
SRC_PATH      = "../data/csvs/vqa.csv"
IMAGE_DIR     = "../data/curated_images"
DEST_PATH     = "../data/csvs/preds_vilt_ft.csv"
BASE_MODEL    = "dandelin/vilt-b32-finetuned-vqa"
USE_FINETUNED = True
SEED          = 7
SAMPLE_SIZE   = 10000
MAX_LENGTH    = 128
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_processor():
    if USE_FINETUNED:
        print(f"Loading finetuned model with LoRA adapters from {MODEL_DIR}")
        processor = ViltProcessor.from_pretrained(MODEL_DIR, use_fast=True)
        base_model = ViltForQuestionAnswering.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    else:
        print(f"Loading baseline ViLT from {BASE_MODEL}")
        processor = ViltProcessor.from_pretrained(BASE_MODEL, use_fast=True)
        model = ViltForQuestionAnswering.from_pretrained(BASE_MODEL)

    model.to(DEVICE)
    model.eval()
    return processor, model

def run_inference(processor, model, df_sample, output_path):
    with open(output_path, mode="w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["image_name", "question", "answer", "generated_answer"])

        for fn, question, answer in tqdm(
            df_sample[["image_name", "question", "answer"]].itertuples(index=False),
            total=len(df_sample),
            desc="Running ViLT VQA Inference"
        ):
            img_path = os.path.join(IMAGE_DIR, fn)
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(
                    images=image,
                    text=question,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch.no_grad():
                    out = model(**inputs)
                    pred_id = out.logits.softmax(dim=1).argmax(dim=1).item()
                    pred = model.config.id2label[pred_id].strip().lower()

            except Exception:
                pred = ""

            writer.writerow([fn, question, answer, pred])

def main():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(SRC_PATH)
    df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED).reset_index(drop=True)
    processor, model = load_model_and_processor()
    run_inference(processor, model, df_sample, DEST_PATH)
    print(f"Saved predictions to {DEST_PATH}")

if __name__ == "__main__":
    main()