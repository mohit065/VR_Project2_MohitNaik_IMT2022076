# [VR Mini Project 2](https://github.com/mohit065/VR_Project2_MohitNaik_IMT2022076) : Multimodal Visual Question Answering and Finetuning with LoRA

Graded mini-project for the AIM825 Visual Recognition course. Involves creating a Visual Question Answering (VQA) dataset using the Amazon Berkeley Objects (ABO) dataset, evaluating baseline models such as BLIP and ViLT, fine-tuning using Low-Rank Adaptation (LoRA), and assessing performance using standard metrics.

This readme only contains the steps to run the code. All other information regarding the project is available in the [report](report.pdf).

## Steps to Run

Clone the repository and add the datasets so that the directory structure looks as follows:

```none
📂data
 ┣ 📂images
 ┃  ┣ 📂metadata
 ┃  ┗ 📂small
 ┣ 📂listings
 ┃  ┗ 📂metadata
 ┣ 📂blip_ft
 ┣ 📂vilt_ft
 ┣ 📂csvs
 ┗ 📂curated_images
📂scripts
```

Ensure you have `python 3.9` or above. To install required libraries, run

```none
pip install -r requirements.txt
```

The scripts folder contains all the code.

- `curate_data.py`: Filters images and metadata. Saves the images to `data/curated_images` and the metadata to `data/csvs/curated.csv`.
  
- `make_vqa.py`: Makes the VQA dataset from the curated images and metadata. Saves it to `data/csvs/vqa.csv`.
  
- `finetune_blip.py`: Finetunes `salesforce/blip-vqa-base` on the VQA dataset. Saves model to `data/blip_ft`. The finetuned model is also put on HuggingFace Hub as `pratster/salesforce_blip_fine_tuned`.
  
- `finetune_vilt.py`: Finetunes `dandelin/vilt-b32-finetuned-vqa` on the VQA dataset. Saves model to `data/vilt_ft`.
  
- `infer_blip.py`: Inference with vanilla/finetuned BLIP model on a subset of the VQA data. Saves predictions to `data/csvs/preds_blip.csv` or `data/csvs/preds_blip_ft.csv`.
  
- `infer_blip.py`: Inference with vanilla/finetuned ViLT model on a subset of the VQA data. Saves predictions to `data/csvs/preds_vilt.csv` or `data/csvs/preds_vilt_ft.csv`.
  
- `eval.py`: Evaluates performance for any set of predictions and displays evaluation metrics.

The `inference.py` file is for submission and supports only the vanilla/finetuned BLIP model. To run it, use the command

```python
python inference.py --image_dir data/curated_images --csv_path data/csvs/vqa.csv
```

## Authors

- [IMT2022017 Prateek Rath](https://github.com/prateek-rath)
- [IMT2022076 Mohit Naik](https://github.com/mohit065)
- [IMT2022519 Vedant Mangrulkar](https://github.com/MVedant21)

---
