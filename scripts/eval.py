import logging
import Levenshtein
import pandas as pd
from statistics import mean
from bert_score import score as bert_score
from nltk.translate.meteor_score import single_meteor_score

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def main():

    PRED_CSV = '../data/csvs/preds_blip_ft.csv'

    # Load data
    df = pd.read_csv(PRED_CSV, dtype=str)
    df = df.fillna('')

    # Exact match accuracy
    print("Computing Exact Match Accuracy...")
    exacts = (df['answer'].str.strip() == df['generated_answer'].str.strip()).astype(float)
    exact_acc = exacts.mean()
    print(f"EXACT ACCURACY : {exact_acc:.4f}\n")

    # Substring match accuracy
    print("Computing Substring Match Accuracy...")
    subs = df.apply(
        lambda row: (row['generated_answer'].strip() in row['answer'].strip()) or
        (row['answer'].strip() in row['generated_answer'].strip()), axis=1
    ).astype(float)
    sub_acc = subs.mean()
    print(f"SUBSTRING ACCURACY : {sub_acc:.4f}\n")

    # Levenshtein similarity
    print("Computing Levenshtein Similarity...")
    levs = [Levenshtein.ratio(a, p) for a, p in zip(df['answer'], df['generated_answer'])]
    lev_mean = mean(levs)
    print(f"LEVENSHTEIN SIMILARITY : {lev_mean:.4f}\n")

    # Meteor score
    print("Computing Meteor Score...")
    meteors = [single_meteor_score(a.split(), p.split()) for a, p in zip(df['answer'], df['generated_answer'])]
    meteor_mean = mean(meteors)
    print(f"METEOR SCORE : {meteor_mean:.4f}\n")

    # BERTScore F1
    print("Computing BERTScore F1...")
    _, _, F = bert_score(df['generated_answer'].tolist(), df['answer'].tolist(), lang='en', rescale_with_baseline=True)
    bert_f1 = F.mean().item()
    print(f"BERT F1 : {bert_f1:.4f}")

if __name__ == '__main__':
    main()