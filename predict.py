import numpy as np
from PIL import Image 
from tensorflow.keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 

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
        





