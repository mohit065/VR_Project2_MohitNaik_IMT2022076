import json
import glob
import pandas as pd
from tqdm import tqdm

SRC_PATH = '../data/images/metadata/images.csv'
DEST_PATH = '../data/curated.csv'
SEED = 42
N_TOTAL = 10000

images_df = pd.read_csv(SRC_PATH)
metadata_lookup = {}

# --- Helper functions ---
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
        deduped_keywords = [k for k in keywords if not (k in seen or seen.add(k))]
        return ', '.join(deduped_keywords)
    return None

def get_metadata(image_id, field):
    return metadata_lookup.get(image_id, {}).get(field, None)

# --- Build metadata lookup ---
json_files = sorted(glob.glob('../data/listings/metadata/listings_*.json'))

for file in tqdm(json_files, desc="Parsing listings", unit="file"):
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
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

# --- Apply metadata ---
images_df['name'] = images_df['image_id'].apply(lambda x: get_metadata(x, 'name'))
images_df['product_type'] = images_df['image_id'].apply(lambda x: get_metadata(x, 'product_type'))
images_df['color'] = images_df['image_id'].apply(lambda x: get_metadata(x, 'color'))
images_df['keywords'] = images_df['image_id'].apply(lambda x: get_metadata(x, 'keywords'))

# --- Filtering ---
images_df = images_df[~images_df['image_id'].isin(['518Dk4FOzZL', '719hoe+OvIL', '71Qbh8wmhnL'])]
images_df.dropna(subset=['name', 'product_type', 'color', 'keywords'], inplace=True)
is_ascii = lambda text: isinstance(text, str) and text.isascii()
for col in ['name', 'product_type', 'color']:
    images_df = images_df[images_df[col].apply(is_ascii)]

for col in ['name', 'product_type', 'color', 'keywords']:
    images_df[col] = images_df[col].str.lower()

images_df.drop(columns=['image_id', 'height', 'width'], errors='ignore', inplace=True)

non_case_df = images_df[images_df['product_type'] != 'cellular_phone_case']
case_df = images_df[images_df['product_type'] == 'cellular_phone_case']
case_sampled = case_df.sample(n=min(N_TOTAL - len(non_case_df), len(case_df)), random_state=0)
filtered_df = pd.concat([non_case_df, case_sampled], ignore_index=True)
filtered_df = filtered_df.sort_values(by='path').reset_index(drop=True)
print(f"Final filtered dataset size: {len(filtered_df)}")

# --- Save output ---
filtered_df.to_csv(DEST_PATH, index=False)
print(f"Saved {len(filtered_df)} entries to {DEST_PATH}")