from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_io as tfio
#from tensorflow.keras import Sequential, layers, models, optimizers
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.layers.experimental.preprocessing import Rescaling


def train_val_test_split(X, y):
    X_bis, X_test, y_bis, y_test = train_test_split(X, y, test_size=0.15)
    X_train, X_val, y_train, y_val = train_test_split(X_bis,
                                                      y_bis,
                                                      test_size=0.2)
    return X_train, y_train, X_val, y_val, X_test, y_test


AUTOTUNE = tf.data.experimental.AUTOTUNE
#BATCH_SIZE = 32  # à définir (32)
#BUFFER_SIZE = 32  # à définir (len(file_path))
#TARGET_HEIGHT = 128  #à définir (hauteur de l'image redimensionnée)
#TARGET_WIDTH = 128  #à définir (largeur de l'image redimensionnée)

def extract_img(file_path,
                label,
                target_height=128,
                target_width=128):
    dcm_file = tf.io.read_file(file_path)
    img = tfio.image.decode_dicom_image(dcm_file)
    img = tf.image.resize_with_pad(img, target_height, target_width)
    img = tf.squeeze(img, axis=0)
    return img, label


def prepare_ds(ds, target_height=128,
                target_width=128, buffer_size=32, batch_size=32):
    ds = ds.map(extract_img(target_height, target_width),
                num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds
