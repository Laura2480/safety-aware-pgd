import hashlib
import os
import time
from datetime import datetime
import re

from numpy.linalg import lstsq
from numpy.polynomial.polynomial import Polynomial, polyvander2d
import numpy as np
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import xml.etree.ElementTree as ET
import torch

max_calls_per_minute = 1000

def build_image_path(category, folder, filename, path_base):
        path = f"{path_base}/{category}/{folder}/{filename}"
        path = "\\\\?\\" + os.path.abspath(path)
        return path

def dall_e_clip_score(prompt,category,folder,filename,prompter,path_base):
    path=f"{path_base}/{category}/{folder}/{filename}"
    path="\\\\?\\" + os.path.abspath(path)
    img=Image.open(path)
    similarity=prompter.measure_similarity(prompt,img)
    return similarity

def sd_clip_score(prompt,filename,prompter):
    path=f"../data/images/open_ai_policy_violation/dalle3X/{filename}.png"
    img=Image.open(path)
    similarity=prompter.measure_similarity(prompt,img)
    return similarity

def dall_e_moderation_image(category,folder,filename,prompter,path_base):
    path=f"{path_base}/{category}/{folder}/{filename}"
    path = "\\\\?\\" + os.path.abspath(path)
    img=Image.open(path)
    features_image = prompter.get_target_feature(target_images=[img]).to(prompter.device)
    return torch.sigmoid(prompter.moderation_model(features_image)).sum().detach().to('cpu').numpy()

def sd_moderation_image(filename,prompter):
    path=f"../data/images/open_ai_policy_violation/dalle3X/{filename}.png"
    img=Image.open(path)
    features_image = prompter.get_target_feature(target_images=[img]).to(prompter.device)
    return torch.sigmoid(prompter.moderation_model(features_image)).sum().detach().to('cpu').numpy()

def moderation_apply(prompter, prompt):
    return prompter.moderation(prompt, score=True)

def process_batch(df_batch, prompter, prompt_column):
    with ThreadPoolExecutor() as executor:
        # Applica la funzione in parallelo a ogni prompt con tqdm per il progresso
        results = list(tqdm(executor.map(lambda prompt: moderation_apply(prompter, prompt), df_batch[prompt_column]), total=len(df_batch[prompt_column])))
    return results

def get_manual_rating_from_xmp(xmp_file):
    try:
        if not os.path.exists(xmp_file):
            return None

        # Parse del file XMP come XML
        tree = ET.parse(xmp_file)
        root = tree.getroot()

        # Definisci i namespace da utilizzare per cercare i tag specifici
        namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'xmp': 'http://ns.adobe.com/xap/1.0/'
        }

        # Trova il nodo che contiene il rating (campo xmp:Rating)
        description_elem = root.find('.//rdf:Description', namespaces)
        if description_elem is not None:
            rating = description_elem.get('{http://ns.adobe.com/xap/1.0/}Rating')
            return rating

        return 0
    except Exception as e:
        print(f"Errore durante la lettura del file XMP: {e}")
        return 0

def extract_xmp_rating_from_image(image):
    # Check if XMP metadata exists
    if 'XML:com.adobe.xmp' in image.info:
        xmp_data = image.info['XML:com.adobe.xmp']

        # Parse the XML data
        try:
            root = ET.fromstring(xmp_data)

            # Define the namespace for XMP
            namespaces = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'xmp': 'http://ns.adobe.com/xap/1.0/'
            }

            # Find the rating element
            description = root.find('.//rdf:Description', namespaces)
            rating = description.get('{http://ns.adobe.com/xap/1.0/}Rating')

            # Return the rating, converted to int if possible
            return int(rating) if rating is not None else None
        except ET.ParseError:
            print("Error parsing XMP metadata.")
            return None
    else:
        print("No XMP metadata found.")
        return None

def get_manual_rating(row,main_folder):
    if row['image_filename']:
        xmp_file_filepath = os.path.join(main_folder, row['category'], row['prompt_folder'], row['image_filename'] )
        xmp_file_filepath = "\\\\?\\" + os.path.abspath(xmp_file_filepath)
        # Access metadata
        #metadata = image.info
        if os.path.exists(xmp_file_filepath):
            image = Image.open(xmp_file_filepath)
            rating = extract_xmp_rating_from_image(image)
            #rating = get_manual_rating_from_xmp(xmp_file_filepath.replace('.png' ,'.xmp'))
            return rating

    return None

def sanitize_prompt(prompt):
    sanitized = re.sub(r'[^\w\s-]', '', prompt)  # Remove special characters except hyphens
    sanitized = re.sub(r'\s+', '_', sanitized).strip()  # Replace spaces with underscores
    return sanitized

def convert_timestamp(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def polynomial_regression(x, y, degree=2):
    coefs = Polynomial.fit(x, y, degree)
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = coefs(x_smooth)
    return x_smooth, y_smooth

def spline_interpolation(x, y, order=2):
    # Ordina i dati per x
    sorted_indices = np.argsort(x)
    x_sorted = np.array(x)[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]

    # Rimuovi duplicati (se presenti) facendo la media dei valori di y
    x_unique, indices = np.unique(x_sorted, return_index=True)
    y_unique = [np.mean(y_sorted[x_sorted == xi]) for xi in x_unique]

    # Applicazione dell'interpolazione spline cubica
    spline = make_interp_spline(x_unique, y_unique, k=order)  # k=3 significa spline cubica
    x_smooth = np.linspace(min(x_unique), max(x_unique), 100)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

def polyfit2d(x, y, z, degree=3):
    vander = polyvander2d(x, y, [degree, degree])
    coeffs, _, _, _ = lstsq(vander, z, rcond=None)
    return coeffs

# Funzione per valutare il polinomio 2D
def polyval2d(x, y, coeffs, degree=3):
    vander = polyvander2d(x, y, [degree, degree])
    z = np.dot(vander, coeffs)
    return z

def unique_image_filename(target_prompt):
    timestamp = str(time.time())
    unique_string = f"{target_prompt}_{timestamp}"
    unique_hash = hashlib.sha256(unique_string.encode()).hexdigest()
    image_filename = f'{unique_hash[0:20]}'
    return image_filename

#Usage
'''
for col in ['original_prompt','unsafe_prompt','revised_safe','soft_prompt']:
    valori_unici = df[col].unique()
    risultati = {valore: moderation_apply(prompter,valore)[0] for valore in valori_unici}
    df[col+'_flagged'] = df[col].map(risultati)
'''

'''
df['dall_e_clip_score'] = df.apply(
    lambda row: dall_e_clip_score(row['original_prompt'], row['category'], row['prompt_folder'], row['image_filename'],prompter,base_path) if row['dall_e_clip_score_extended'] is None else row['dall_e_clip_score'],
    axis=1
)

df['dall_e_clip_score_extended'] = df.apply(
    lambda row: dall_e_clip_score(row['unsafe_prompt'], row['category'], row['prompt_folder'], row['image_filename'],prompter,base_path) if row['dall_e_clip_score_extended'] is None else row['dall_e_clip_score_extended'],
    axis=1
)

df['sd_clip_score'] = df.apply(
    lambda row: sd_clip_score(row['original_prompt'], row['sd_image_filename'],prompter) if row['sd_clip_score'] is None else row['sd_clip_score'],
    axis=1
)

df['sd_clip_score_extended'] = df.apply(
    lambda row: sd_clip_score(row['unsafe_prompt'], row['sd_image_filename'],prompter) if row['sd_clip_score_extended'] is None else row['sd_clip_score_extended'],
    axis=1
)

df['dall_e_clip_image_moderation'] = df.apply(
    lambda row: dall_e_moderation_image(row['category'], row['prompt_folder'], row['image_filename'],prompter,base_path) if row['dall_e_clip_image_moderation'] is None else row['dall_e_clip_image_moderation'],
    axis=1
)

df['sd_clip_image_moderation'] = df.apply(
    lambda row: sd_moderation_image(row['sd_image_filename'] ) if row['sd_clip_image_moderation'] is None else row['sd_clip_image_moderation'],
    axis=1
)

df[['dall_e_clip_score', 'dall_e_clip_score_extended', 'dall_e_clip_image_moderation']] = df[['dall_e_clip_score', 'dall_e_clip_score_extended', 'dall_e_clip_image_moderation']].astype(float)

df[['sd_clip_score', 'sd_clip_score_extended', 'sd_clip_image_moderation']] = df[['sd_clip_score', 'sd_clip_score_extended', 'sd_clip_image_moderation']].astype(float)

'''