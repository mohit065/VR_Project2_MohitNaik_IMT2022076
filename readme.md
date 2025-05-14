# [VR Mini Project 2](https://github.com/mohit065/VR_Project2_MohitNaik_IMT2022076) : Multimodal Visual Question Answering and Finetuning with LoRA

## Index

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing and Curation](#preprocessing-and-curation)
- [Baseline VQA Models](#baseline-vqa-models)
- [Finetuning](#finetuning)
- [Results and Experiments](#results-and-experiments)
- [Observations and Challenges](#observations-and-challenges)
- [Steps to Run](#steps-to-run)
- [Contributions](#contributions)

---

## Introduction

This assignment involves creating a Visual Question Answering (VQA) dataset using the Amazon Berkeley Objects (ABO) dataset, evaluating baseline models such as CLIP, BLIP and ViLT, fine-tuning using Low-Rank Adaptation (LoRA), and assessing performance using standard metrics.

---

## Dataset

We are using the [Amazon Berkeley Objects Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), which contains around 150k Amazon product listings with multilingual metadata and around 400k catalog images.
- The listings metadata contains things like the product name, category, brand, color, keywords associated with the product etc.
- The images are variable-sized catalog images with a single item on a white background.

---

## Preprocessing and Curation

Around 85% of the images are mobile phone cases and covers. The rest 15% come from a variety of categories such as clothing, shoes, home items, electronics, furniture, accessories, kitchen and groceries etc. We performed the following preprocessing tasks on the images and the listings:
- Ignored all products which have non-English metadata, followed by removing those with non-ASCII metadata.
- Ignored all products which do not have metadata for product name, type, color and keywords.
- Considered all remaining products which are not mobile phone covers, and then considered mobile phone covers to bring the total number of images to 10000. This was done to offset the huge imbalance in the number of mobile phone cover listings compared to other products.
- Resized all images to 256*256 for consistency, sorted by filename and saved the images to a separate folder, while saving the filename, name, product_type, color and keywords in a csv.

To create the VQA dataset, we used the Google Gemini API to generate question answer pairs for each image. We used the `gemini-1.5-flash` model, to which we passed the product image, along with its metadata, and a prompt to generate 5 distinct questions per product. The prompt also encourages the model to generate a mix of questions; ones which can be directly answered by looking at the image such as the product color or product name, and ones which require knowledge and thinking such as product functionality or fine details.

After the entire dataset processing and VQA generation process, we get a set of 50000 question-answer pairs, 5 pairs each for 10000 selected images.

## Baseline VQA models

## Finetuning

## Results and Experiments

## Observations and Challenges

## Steps to Run

Clone the repository and add the datasets so that the directory structure looks as follows:

```none
📂data
 ┣ 📂images
 ┃ ┣ 📂metadata
 ┃ ┗ 📂small
 ┣ 📂listings
 ┗ ┗ 📂metadata
📂scripts
```

Ensure you have `python 3.10`. To install required libraries, run

```none
pip install -r requirements.txt
```

To create the curated dataset, run `curator.py`. It will save the output in `data/curated.csv`, along with copying the selected resized images in `data/curated_images`.

To create the VQA dataset, run `vqa_maker.py`. It will save the output in `data/vqa.csv`.

---

## Contributions

- IMT2022017 Prateek Rath : 
- IMT2022076 Mohit Naik : 
- IMT2022519 Vedant Mangrulkar : 

---
