import os
import numpy as np
import pandas as pd
from pydicom import dcmread
#from pydicom.data import get_testdata_file

local_path = os.path.join('..', 'raw_data')
gdrive_path = '/content/drive/MyDrive/'


def clear_id_image(image_id):
    image_id = image_id.replace('_image', '')
    return image_id

def clear_id_study(study_id):
    study_id = study_id.replace('_study', '')
    return study_id


def merged_csv_df(path):
    ''' if path : 'local_path' = os.path.join('..', 'raw_data')
        if path : 'gdrive_path' = '/content/drive/MyDrive/'
        OR defined as another variable elsewhere
    '''
    image = pd.read_csv(os.path.join(path, 'train_image_level.csv'))
    study = pd.read_csv(os.path.join(path, 'train_study_level.csv'))
    df_image = image.copy()
    df_study = study.copy()
    df_image['image_id'] = df_image['id'].apply(clear_id_image)
    df_study['StudyInstanceUID'] = df_study['id'].apply(clear_id_study)
    df = (df_image.drop(columns='id')).merge(df_study.drop(columns='id'),
                                             on='StudyInstanceUID',
                                             how='left')
    return df


def csv_simple_df(path):
    ''' if path : 'local_path' = os.path.join('..', 'raw_data')
        if path : 'gdrive_path' = '/content/drive/MyDrive/'
        OR defined as another variable elsewhere
        Returns: Dataframe with 'image_id', studyID, and label categories (targets)
    '''
    df = merged_csv_df(path)
    simple_df = df.drop(columns=['boxes', 'label'])
    return simple_df


def images_path_df(image_path):
    images_list = os.listdir(image_path)
    images_path=[]
    images_id_list =[]
    for i in images_list:
        path = (image_path + "/" + i)
        images_path.append(path)
        element = i.replace(".dcm", "")
        images_id_list.append(element)
    images_path_df = pd.DataFrame(images_path, columns=['images_path'])
    images_path_df['image_id'] = images_id_list
    return images_path_df


def get_tensorflow_ready(dataframe1, dataframe2):
    tensorflow_df = dataframe1.merge(dataframe2, on='image_id',
                                         how='inner').set_index('image_id')
    file_paths = tensorflow_df["images path"].values
    labels = tensorflow_df[[
    'Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance',
    'Atypical Appearance']].values
    return file_paths, labels


def load_image_data(loading_method, columns = ['Rows', 'Columns']):
    ''' loading_method: 'colab' or 'direct' ==> loads the informations of ALL the images in the given path
    Columns names must be .dcm pictures parameters (ex: Rows, Columns, pixel_array, ...)
    Returns: DataFrame with first column = name of .dcm file (without '.dcm') named 'image_id', and other columns as arguments
    '''
    if loading_method == 'colab':
        data_path = gdrive_path
    elif loading_method == 'direct':
        data_path = local_path
    file_names = [f for f in os.listdir(data_path) if f.endswith('.dcm')]
    image_id = [key_name.replace('.dcm', '') for key_name in file_names]
    image_data = pd.DataFrame(columns=[['image_id'] + columns])
    for ids, file in zip(image_id, file_names):
        ds = dcmread(os.path.join(data_path, file))
        image_data = image_data.append(
            ids, {field: ds[field].value
                  for field in columns},
            ignore_index=True)
    return image_data