import os
import numpy as np
import pandas as pd
from pydicom import dcmread
#from pydicom.data import get_testdata_file

local_path = os.path.join('..', 'raw_data')
gdrive_path = '/content/drive/MyDrive/New_train_subset/'

def get_df_from_csv():
    image = pd.read_csv(os.path.join(local_path, 'train_image_level.csv'))
    study = pd.read_csv(os.path.join(local_path, 'train_study_level.csv'))
    df_image = image.copy()
    df_study = study.copy()
    return df_image, df_study

def clear_id_image(image_id):
    image_id = image_id.replace('_image', '')
    return image_id


def clear_id_study(study_id):
    study_id = study_id.replace('_study', '')
    return study_id

def merged_df():
    df_image, df_study = get_df_from_csv()
    df_image['image_id'] = df_image['id'].apply(clear_id_image)
    df_study['StudyInstanceUID'] = df_study['id'].apply(clear_id_study)
    df = (df_image.drop(columns='id')).merge(df_study.drop(columns='id'),
                                             on='StudyInstanceUID',
                                             how='left')
    return df

def get_simple_df():
    df = merged_df()
    simple_df = df.drop(columns=['boxes', 'label'])
    return simple_df


def load_image_data(loading_method):
    if loading_method == 'colab':
        data_path = gdrive_path
    elif loading_method == 'direct':
        data_path = local_path
    file_names = [f for f in os.listdir(data_path) if f.endswith('.dcm')]
    image_id = [key_name.replace('.dcm', '') for key_name in file_names]
    image_data = []
    for ids, file in zip(image_id, file_names):
        ds = dcmread(os.path.join(data_path, file))
        image_data.append([
            ids, ds.StudyInstanceUID, ds.PatientSex, ds.ImagerPixelSpacing,
            ds.PhotometricInterpretation, ds.Rows, ds.Columns, ds.pixel_array
        ])
    return image_data