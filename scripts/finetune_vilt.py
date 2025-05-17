import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from bert_score import score as bertscore_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import ViltProcessor, ViltForQuestionAnswering, get_scheduler

SRC_PATH       = '../data/csvs/vqa.csv'
IMAGE_DIR      = '../data/curated_images'
DEST_DIR       = '../data/vilt_ft'
MODEL_NAME      = 'dandelin/vilt-b32-finetuned-vqa'
BATCH_SIZE      = 16
EVAL_BATCH_SIZE = 32
N_EPOCHS        = 3
LEARNING_RATE   = 5e-5
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.1
MAX_LENGTH      = 128
WARMUP_STEPS    = 0

def create_dataset(df, image_dir, label2id):
    class VQADataset(Dataset):
        def __init__(self, dataframe):
            self.image_dir = image_dir
            self.entries = []
            self.label2id = label2id
            self.unk_id = label2id.get('other', label2id.get('unknown', 0))
            for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Verifying images"):
                img_path = os.path.join(self.image_dir, str(row['image_name']))
                ans = str(row['answer'])
                if os.path.exists(img_path):
                    ans_id = self.label2id.get(ans, self.unk_id)
                    image = Image.open(img_path).convert('RGB')
                    self.entries.append({
                        "image": image,
                        "question": str(row['question']),
                        "answer": ans_id
                    })
                else:
                    print(f"Warning: Missing {img_path}")
            if not self.entries:
                raise RuntimeError('No valid entries found.')

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, idx):
            return self.entries[idx]
    return VQADataset(df)

def vqa_collate_fn(batch, processor, base_model, device):
    images    = [item["image"]    for item in batch]
    questions = [item["question"] for item in batch]
    ans_ids   = [item["answer"]   for item in batch]

    enc = processor(
        images=images,
        text=questions,
        truncation=True,
        padding='longest',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

    batch_size = len(ans_ids)
    num_labels = base_model.config.num_labels
    labels = torch.zeros((batch_size, num_labels), dtype=torch.float)
    for i, ans_id in enumerate(ans_ids):
        labels[i, ans_id] = 1.0

    enc['labels'] = labels
    return {k: v.to(device) for k, v in enc.items()}

def load_model_and_processor():
    print(f"Loading {MODEL_NAME}…")
    processor = ViltProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    base_model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME)
    print(f"{MODEL_NAME} loaded successfully.")
    return processor, base_model

def apply_lora(base_model):
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "key", "value"],
        lora_dropout=LORA_DROPOUT,
        bias="none"
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model

def train_loop(model, train_loader, optimizer, scheduler, accelerator, epoch):
    model.train()
    total_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch} train")
    for step, batch in enumerate(train_bar, 1):
        out = model(**batch)
        loss = out.loss
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        train_bar.set_postfix(train_loss=total_loss / step)
    return total_loss / len(train_loader)

def eval_loop(model, eval_loader, id2label, epoch):
    model.eval()
    preds, refs = [], []
    eval_bar = tqdm(eval_loader, desc=f"Epoch {epoch} eval")
    for batch in eval_bar:
        with torch.no_grad():
            out = model(**{k: batch[k] for k in ['input_ids', 'attention_mask', 'pixel_values']})
            pred_ids = out.logits.argmax(dim=-1)
            label_ids = batch['labels'].argmax(dim=-1)

            preds.extend([id2label.get(i.item(), "unknown") for i in pred_ids])
            refs.extend([id2label.get(i.item(), "unknown") for i in label_ids])

    _, _, F1 = bertscore_score(preds, refs, lang="en", rescale_with_baseline=True)
    avg_f1 = F1.mean().item()
    return avg_f1

def main():
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    print(f"Using {accelerator.state.num_processes} GPU(s), fp16")

    processor, base_model = load_model_and_processor()
    label2id = base_model.config.label2id
    id2label = {v: k for k, v in label2id.items()}

    model = apply_lora(base_model)

    df = pd.read_csv(SRC_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=7)

    train_ds = create_dataset(train_df, IMAGE_DIR, label2id)
    val_ds = create_dataset(val_df, IMAGE_DIR, label2id)
    print(f"train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda batch: vqa_collate_fn(batch, processor, base_model, device)
    )
    eval_loader = DataLoader(
        val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=lambda batch: vqa_collate_fn(batch, processor, base_model, device)
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * N_EPOCHS
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )
    accelerator.init_trackers("vilt-vqa")

    print("Starting finetuning…")
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_loop(model, train_loader, optimizer, scheduler, accelerator, epoch)
        avg_f1 = eval_loop(model, eval_loader, id2label, epoch)

        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, eval_bertscore_f1={avg_f1:.4f}\n")
            model.save_pretrained(DEST_DIR)
            processor.save_pretrained(DEST_DIR)
            print(f"Model saved to {DEST_DIR}.")

    print("Finetuning complete.")

if __name__ == "__main__":
    main()