from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz

import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}




@app.get("/predict")
def predict(radio_img):
    return {}
    


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  fp = "cnn_pneu_vamp_model.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()
st.write("""
# Chest Radiography Classification [Negative for Pneumonia
                                    Typical Appearance
                                    Intermediate Appearance 
                                    Atypical Appearance
                                    ]
by The COVID-19 Warriors Team ;)

Le Wagon Marseile : Data Science Bach #610
""")




temp = st.file_uploader("Upload Your Chest Radiograph Bellow")
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
  st.text("Oops! The image format is not recognised :( Please Try Another Format")

else:

 

  covid_img = image.load_img(temp_file.name, target_size=(500, 500),color_mode='grayscale')

  # Preprocessing the image
  pp_covid_img = image.img_to_array(covid_img)
  pp_covid_img = pp_hardik_img/255
  pp_covid_img = np.expand_dims(pp_covid_img, axis=0)

  #predict
  covid_preds= cnn.predict(pp_covid_img)
  if covid_preds>= 0.5:
    out = ('I am {:.2%} percent confirmed that this is a Pneumonia case'.format(covid_preds[0][0]))
  
  else: 
    out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(1-covid_preds[0][0]))

  st.success(out)
  
  image = Image.open(temp)
  st.image(image,use_column_width=True)
        















    # # create a datetime object from the user provided datetime
    # pickup_datetime = "2021-05-30 10:12:00"
    # pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # # localize the user datetime with NYC timezone
    # eastern = pytz.timezone("US/Eastern")
    # localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # # localize the datetime to UTC
    # utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)


    # formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # key_ = "2013-07-06 17:18:00.000000119"


    # X_pred = pd.DataFrame(dict(
    #     key=[key_],
    #     pickup_datetime=[formatted_pickup_datetime],
    #     pickup_longitude=[float(pickup_longitude)],
    #     pickup_latitude=[float(pickup_latitude)],
    #     dropoff_longitude=[float(dropoff_longitude)],
    #     dropoff_latitude=[float(dropoff_latitude)],
    #     passenger_count=[int(passenger_count)]))



    # # user_input = {"key" : key,
    # # "pickup_datetime": pickup_datetime,
    # # "pickup_longitude": float(pickup_longitude),
    # # "pickup_latitude": float(pickup_latitude),
    # # "dropoff_longitude": float(dropoff_longitude),
    # # "dropoff_latitude": float(dropoff_latitude),
    # # "passenger_count": int(passenger_count)}
    # # X_pred = pd.DataFrame(data=user_input)

    # pipeline = joblib.load('model.joblib')
    # results = pipeline.predict(X_pred)
    # pred = float(results[0])

    # return {"prediction" : pred}

    