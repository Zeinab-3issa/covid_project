import numpy as np
import pandas as pd

def get_df_from_csv():
    image = pd.read_csv('../raw_data/train_image_level.csv')
    study = pd.read_csv('../raw_data/train_study_level.csv')
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
