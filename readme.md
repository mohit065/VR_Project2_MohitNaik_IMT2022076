# [VR Mini Project 2](https://github.com/mohit065/VR_Project2_MohitNaik_IMT2022076) : Multimodal Visual Question Answering and Finetuning with LoRA

Graded mini-project for the AIM825 Visual Recognition course. Involves creating a Visual Question Answering (VQA) dataset using the Amazon Berkeley Objects (ABO) dataset, evaluating baseline models such as BLIP and ViLT, fine-tuning using Low-Rank Adaptation (LoRA), and assessing performance using standard metrics.

This readme only contains the steps to run the code. All other information regarding the project is available in the [report](report.pdf).

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

All notebooks are present in the `notebooks` folder. To create the curated dataset at `data/curated.csv`, run `curator_data.ipynb`. This will also save the selected resized images in `data/curated_images`. To create the VQA dataset at `data/vqa.csv`, run `make_vqa.ipynb`.

To make the baseline inference predictions at `data/preds_blip.csv` and `data/preds_vilt.csv`, run `infer_blip.ipynb` and `infer_vilt.ipynb`. To evaluate the predictions, run `eval.ipynb`.

## Authors

- [IMT2022017 Prateek Rath](https://github.com/prateek-rath)
- [IMT2022076 Mohit Naik](https://github.com/mohit065)
- [IMT2022519 Vedant Mangrulkar](https://github.com/MVedant21)

---
