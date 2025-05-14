import os
import glob
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Config
SRC_PATH = '../data/images/metadata/images.csv'
IMAGE_SRC_DIR = '../data/images/small'
IMAGE_DEST_DIR = '../data/curated_images'
DEST_PATH = '../data/curated.csv'

N_TOTAL = 10000
TARGET_SIZE = (256, 256)

# Read images metadata CSV
images_df = pd.read_csv(SRC_PATH)
metadata_lookup = {}

# Helper functions
def extract_field(data, key, inner_key='value'):
    if isinstance(data.get(key), list) and data[key]:
        first = data[key][0]
        if 'language_tag' in first and not first['language_tag'].startswith('en_'):
            return None
        return first.get(inner_key, None)
    return None

def extract_keywords(data):
    if isinstance(data.get('item_keywords'), list):
        keywords = [
            k['value'].strip().lower()
            for k in data['item_keywords']
            if 'language_tag' not in k or k['language_tag'].startswith('en_')
        ]
        seen = set()
        return ', '.join([k for k in keywords if not (k in seen or seen.add(k))])
    return None

def get_metadata(image_id, field):
    return metadata_lookup.get(image_id, {}).get(field)

# Build metadata lookup from JSON listings
json_files = sorted(glob.glob('../data/listings/metadata/listings_*.json'))
for file in tqdm(json_files, desc="Parsing listings", unit="file"):
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                main_id = record.get('main_image_id')
                if main_id:
                    metadata_lookup[main_id] = {
                        'name': extract_field(record, 'item_name'),
                        'product_type': extract_field(record, 'product_type'),
                        'color': extract_field(record, 'color'),
                        'keywords': extract_keywords(record)
                    }
            except json.JSONDecodeError:
                continue

# Attach metadata to images_df
images_df['name'] = images_df['image_id'].apply(lambda x: get_metadata(x, 'name'))
images_df['product_type'] = images_df['image_id'].apply(lambda x: get_metadata(x, 'product_type'))
images_df['color'] = images_df['image_id'].apply(lambda x: get_metadata(x, 'color'))
images_df['keywords'] = images_df['image_id'].apply(lambda x: get_metadata(x, 'keywords'))

# Filter out unwanted IDs and rows missing metadata
images_df = images_df[~images_df['image_id'].isin(['518Dk4FOzZL', '719hoe+OvIL', '71Qbh8wmhnL'])]
images_df.dropna(subset=['name', 'product_type', 'color', 'keywords'], inplace=True)

# only ascii in name/product_type/color
is_ascii = lambda text: isinstance(text, str) and text.isascii()
for col in ['name', 'product_type', 'color']:
    images_df = images_df[images_df[col].apply(is_ascii)]

# lowercase all textual fields
for col in ['name', 'product_type', 'color', 'keywords']:
    images_df[col] = images_df[col].str.lower()

# Balance out phone cases and non-cases
non_case_df = images_df[images_df['product_type'] != 'cellular_phone_case']
case_df     = images_df[images_df['product_type'] == 'cellular_phone_case']
case_sample = case_df.sample(n=min(N_TOTAL - len(non_case_df), len(case_df)))
filtered_df = pd.concat([non_case_df, case_sample], ignore_index=True)

# Sort by original path
filtered_df = filtered_df.sort_values(by='path').reset_index(drop=True)
print(f"Final filtered dataset size: {len(filtered_df)}")

# Derive filename and resize+copy images
os.makedirs(IMAGE_DEST_DIR, exist_ok=True)
filtered_df['filename'] = filtered_df['path'].apply(os.path.basename)

for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Resizing & copying images"):
    src = os.path.normpath(os.path.join(IMAGE_SRC_DIR, row['path']))
    dst = os.path.join(IMAGE_DEST_DIR, row['filename'])
    try:
        with Image.open(src) as img:
            resized = img.resize(TARGET_SIZE, Image.LANCZOS)
            resized.save(dst)
    except FileNotFoundError:
        print(f"Warning: source image not found: {src}")
    except Exception as e:
        print(f"Error processing {src}: {e}")

# Write out curated CSV
output_df = filtered_df[['filename', 'name', 'product_type', 'color', 'keywords']]
output_df.to_csv(DEST_PATH, index=False)
print(f"Saved {len(output_df)} entries to {DEST_PATH}")