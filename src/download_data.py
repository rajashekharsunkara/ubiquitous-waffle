import os
import pandas as pd
import urllib.request
from tqdm import tqdm
import zipfile
import shutil
from pathlib import Path

COSWARA_URL = "https://raw.githubusercontent.com/iiscleap/Coswara-Data/master/combined_data.csv"
COSWARA_AUDIO_BASE = "https://github.com/iiscleap/Coswara-Data/raw/master/Cleaned_data"
ICBHI_BASE_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database"

def download_file(url, filename, description=None):
    """Download file with progress bar"""
    try:
        with tqdm(unit='B', unit_scale=True, desc=description or filename) as pbar:
            urllib.request.urlretrieve(
                url,
                filename,
                reporthook=lambda count, block_size, total_size: pbar.update(block_size)
            )
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def download_coswara():
    """Download Coswara dataset"""
    os.makedirs('data/coswara', exist_ok=True)
    
    print("Downloading Coswara metadata...")
    if download_file(COSWARA_URL, 'data/coswara/metadata.csv'):
        df = pd.read_csv('data/coswara/metadata.csv')
        
        # Download audio files for positive and negative cases
        print("Downloading Coswara audio files...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            if row['covid_status'] in ['positive', 'negative']:
                for sound_type in ['cough-heavy', 'cough-shallow', 'breathing-deep', 'breathing-shallow']:
                    filename = f"{row['id']}_{sound_type}.wav"
                    url = f"{COSWARA_AUDIO_BASE}/{row['id']}/{sound_type}.wav"
                    save_path = f"data/coswara/{row['covid_status']}"
                    os.makedirs(save_path, exist_ok=True)
                    download_file(url, os.path.join(save_path, filename))

def download_icbhi():
    """Download ICBHI dataset"""
    os.makedirs('data/icbhi', exist_ok=True)
    
    print("""
The ICBHI dataset requires registration at:
https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

Please download the following files manually and place them in the data/icbhi directory:
1. ICBHI_final_database_files.zip
2. ICBHI_final_database_annotations.txt

Once downloaded, run this script again to process the files.
""")
    
    # Check if files exist and process them
    if os.path.exists('data/icbhi/ICBHI_final_database_files.zip') and \
       os.path.exists('data/icbhi/ICBHI_final_database_annotations.txt'):
        print("Processing ICBHI dataset...")
        
        # Extract audio files
        with zipfile.ZipFile('data/icbhi/ICBHI_final_database_files.zip', 'r') as zip_ref:
            zip_ref.extractall('data/icbhi/audio')
        
        # Process annotations
        annotations = pd.read_csv('data/icbhi/ICBHI_final_database_annotations.txt', 
                                delimiter='\t', header=None,
                                names=['filename', 'start', 'end', 'crackles', 'wheezes'])
        
        # Organize files by condition
        for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
            source = os.path.join('data/icbhi/audio', row['filename'])
            if row['crackles'] == 1 and row['wheezes'] == 0:
                dest_dir = 'data/icbhi/crackles'
            elif row['crackles'] == 0 and row['wheezes'] == 1:
                dest_dir = 'data/icbhi/wheezes'
            elif row['crackles'] == 1 and row['wheezes'] == 1:
                dest_dir = 'data/icbhi/both'
            else:
                dest_dir = 'data/icbhi/normal'
            
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.exists(source):
                shutil.copy2(source, os.path.join(dest_dir, row['filename']))

def create_labels():
    """Create combined labels file"""
    labels = []
    
    # Process Coswara labels
    if os.path.exists('data/coswara/metadata.csv'):
        coswara_df = pd.read_csv('data/coswara/metadata.csv')
        for _, row in coswara_df.iterrows():
            if row['covid_status'] in ['positive', 'negative']:
                for sound_type in ['cough-heavy', 'cough-shallow', 'breathing-deep', 'breathing-shallow']:
                    filename = f"{row['id']}_{sound_type}.wav"
                    labels.append({
                        'file_name': filename,
                        'dataset': 'coswara',
                        'label': 1 if row['covid_status'] == 'positive' else 0,
                        'condition': 'covid'
                    })
    
    # Process ICBHI labels
    icbhi_annot_path = 'data/icbhi/ICBHI_final_database_annotations.txt'
    if os.path.exists(icbhi_annot_path):
        icbhi_df = pd.read_csv(icbhi_annot_path, delimiter='\t', header=None,
                              names=['file_name', 'start', 'end', 'crackles', 'wheezes'])
        for _, row in icbhi_df.iterrows():
            condition = 'normal'
            if row['crackles'] == 1 and row['wheezes'] == 0:
                condition = 'crackles'
            elif row['crackles'] == 0 and row['wheezes'] == 1:
                condition = 'wheezes'
            elif row['crackles'] == 1 and row['wheezes'] == 1:
                condition = 'both'
            
            labels.append({
                'file_name': row['file_name'],
                'dataset': 'icbhi',
                'label': 0 if condition == 'normal' else 1,
                'condition': condition
            })
    
    # Save labels
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv('data/labels.csv', index=False)
    
    print("\nDataset statistics:")
    print("\nCoswara dataset:")
    coswara_stats = labels_df[labels_df['dataset'] == 'coswara']['label'].value_counts()
    print(f"COVID-negative: {coswara_stats.get(0, 0)}")
    print(f"COVID-positive: {coswara_stats.get(1, 0)}")
    
    print("\nICBHI dataset:")
    icbhi_conditions = labels_df[labels_df['dataset'] == 'icbhi']['condition'].value_counts()
    print(icbhi_conditions)

def prepare_dataset():
    """Download and prepare both datasets"""
    print("Step 1: Downloading Coswara dataset...")
    download_coswara()
    
    print("\nStep 2: Setting up ICBHI dataset...")
    download_icbhi()
    
    print("\nStep 3: Creating labels file...")
    create_labels()

if __name__ == "__main__":
    prepare_dataset()