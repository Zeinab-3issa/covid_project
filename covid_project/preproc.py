from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efnet_preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
#BUFFER_SIZE = 32  # à définir, total lignes dataset ==> ex:(len(file_path))
#TARGET_HEIGHT = 128  #à définir (hauteur de l'image redimensionnée)
#TARGET_WIDTH = 128  #à définir (largeur de l'image redimensionnée)

def prepare_ds(ds, with_labels=True, target_height=128, target_width=128,
               buffer_size=32, batch_size=32,
               data_augmentation=False, transform_parameters=None,
               to_rgb=False, to_vgg=False, to_efnet=False):

    #def create_tensor(X, y=y):
    #    ds = tf.data.Dataset.from_tensor_slices((X, y))
    #    return ds

    def extract_img(file_path, label):
        dcm_file = tf.io.read_file(file_path)
        img = tfio.image.decode_dicom_image(dcm_file)
        img = tf.image.resize_with_pad(img, target_height, target_width)
        if to_rgb == True:
            img = tf.image.grayscale_to_rgb(img)
            if to_vgg == True:
                img = vgg19_preprocess_input(img)
            if to_efnet == True :
                img = efnet_preprocess_input(img)
        img = tf.squeeze(img, axis=0)

        return img, label if with_labels else img

    def augment_img(img,label):
        datagen = ImageDataGenerator()
        new_img = datagen.apply_transform(img, transform_parameters)
        #images = [img]
        #labels = [label]
        #for params in transform_parameters:
        #    new_img = datagen.apply_transform(img, params)
        #    images.append(new_img)
        #    labels.append(label)
        return new_img, label

    def build_augmenter(with_labels=True):
        def augment(img):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            return img

        def augment_with_labels(img, label):
            return augment(img), label

        return augment_with_labels if with_labels else augment

    #ds = create_tensor(X,y)
    ds = ds.map(extract_img,
                num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if data_augmentation == True:
        ds = ds.map(build_augmenter, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)

    return ds
