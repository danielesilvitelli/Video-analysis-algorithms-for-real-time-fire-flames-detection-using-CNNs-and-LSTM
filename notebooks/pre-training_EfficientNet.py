import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
from tensorflow import keras
from keras.models import Model
from keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings('ignore')



dataset = pd.read_csv('/user/dsilvitelli/kerasenv/bin/Tesi Magistrale/Dataset_finale/Vers_base/CSV/MNV2/Urban/Dataset_Urban_img_baseline.csv')

#Qui lo split è stato fatto a mano, quindi esistono già i set predisposti
#Legge train.csv e ricava le colonne interessate
training_data = pd.read_csv("/user/dsilvitelli/kerasenv/bin/Tesi Magistrale/Dataset_finale/Vers_base/CSV/MNV2/Urban/train3.csv", header=0, usecols=[0,1])

training_img_old_path = list(training_data["original_filename_path"])
training_labels = list(training_data["file_class"])

training_img = []
for row in training_img_old_path:
    training_img.append('/user/dsilvitelli/kerasenv/bin' + row[23:])

training_data = pd.DataFrame( {'Images': training_img,'Labels': training_labels})
training_data.Labels = training_data.Labels.astype(str)

#Legge val.csv e ricava le colonne interessate
val_data = pd.read_csv("/user/dsilvitelli/kerasenv/bin/Tesi Magistrale/Dataset_finale/Vers_base/CSV/MNV2/Urban/val3.csv", header=0, usecols=[0,1])

val_img_old_path = list(val_data["original_filename_path"])
val_labels = list(val_data["file_class"])

val_img = []
for row in val_img_old_path:
    val_img.append('/user/dsilvitelli/kerasenv/bin' + row[23:])

val_data = pd.DataFrame( {'Images': val_img,'Labels': val_labels})
val_data.Labels = val_data.Labels.astype(str)

#Legge test.csv e ricava le colonne interessate
test_data = pd.read_csv("/user/dsilvitelli/kerasenv/bin/Tesi Magistrale/Dataset_finale/Vers_base/CSV/MNV2/Urban/test3.csv", header=0, usecols=[0,1])

test_img_old_path = list(test_data["original_filename_path"])
test_labels = list(test_data["file_class"])

test_img = []
for row in test_img_old_path:
    test_img.append('/user/dsilvitelli/kerasenv/bin' + row[23:])

test_data = pd.DataFrame( {'Images': test_img, 'Labels': test_labels})
test_data.Labels = test_data.Labels.astype(str)


######################################################################################


#Calcolo coefficienti class weights
weights_sklearn = compute_class_weight(class_weight = 'balanced', classes = np.unique(np.array(training_labels)), y = np.array(training_labels))
weights = {0: weights_sklearn[0], 1: weights_sklearn[1]}
# print(weights)
# print((np.array(training_labels)==0).mean(), (np.array(training_labels)==1).mean())

from tensorflow.keras.applications.efficientnet import preprocess_input

#Data Augmentation
def preprocess_input_aug(image_noaug):
    image_noaug = image_noaug.astype('uint8')
    transform = A.Compose([ 
        A.HorizontalFlip(p=0.5),
        A.OneOf(
            [
                A.Blur(p=0.01, blur_limit=(3, 7)),
                A.MedianBlur(p=0.01, blur_limit=(3, 7)),
                A.MotionBlur(p=0.01, blur_limit=(3, 7)),
                A.GaussianBlur(p=0.01, blur_limit=(3, 7), sigma_limit=(0.0, 0)),
                A.ZoomBlur(p=0.01, max_factor=(1.0, 1.12), step_factor=(0.01, 0.03)),
                A.Defocus(p=0.01, radius=(1, 4), alias_blur=(0.1, 0.5)),
                A.RingingOvershoot(p=0.01, blur_limit=(7, 15), cutoff=(0.7, 1.57)),
                A.Downscale(p=0.01, scale_min=0.8, scale_max=0.99),
                A.ImageCompression(p=0.01, quality_lower=80, quality_upper=100,
                                    compression_type=0),
                A.JpegCompression(p=0.01, quality_lower=80, quality_upper=100),
                A.GaussNoise(p=0.01, var_limit=(10, 50), per_channel=True, mean=0.0),
                A.MultiplicativeNoise(p=0.01, multiplier=(0.9, 1.1), 
                                        per_channel=True, elementwise=True),
                # A.CLAHE(p=0.01, clip_limit=(1,4), tile_grid_size=(8, 8)),
                # A.Sharpen(p=0.01, alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
                # A.UnsharpMask(p=0.01, blur_limit=(3, 7), sigma_limit=(0.0, 0.0),
                #                 alpha=(0.2, 0.5), threshold=10),
                # A.Emboss(p=0.01, alpha=(0.2, 0.5), strength=(0.2, 0.7)),
                # A.RandomBrightness(p=0.01, limit=(-0.2, 0.2)),
                # A.RandomContrast(p=0.01, limit=(-0.2, 0.2)),
                # A.RandomBrightnessContrast(p=0.01, brightness_limit=(-0.2, 0.2),
                #                             contrast_limit=(-0.2, 0.2), 
                #                             brightness_by_max=True),
                # A.ISONoise(p=0.01, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
                # A.Equalize(p=0.01, mode='cv', by_channels=False),             
                # A.FancyPCA(p=0.01, alpha=0.1),
                # A.RGBShift(p=0.01, r_shift_limit=(-20, 20), 
                #             g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
                # A.RandomGamma(p=0.01, gamma_limit=(80, 120), eps=None),
                # A.RandomToneCurve(p=0.01, scale=0.1),
                # A.SafeRotate(p=0.01, limit=(-90, 90), interpolation=0, 
                #             border_mode=0, value=(0, 0, 0), mask_value=None),
                # A.OpticalDistortion(p=0.01, distort_limit=(-0.3, 0.3),
                #                     shift_limit=(-0.05, 0.05), interpolation=0,
                #                     border_mode=0, value=(0, 0, 0), mask_value=None),
                # A.GridDistortion(p=0.01, num_steps=5, distort_limit=(-0.3, 0.3),
                #                 interpolation=0, border_mode=0, value=(0, 0, 0),
                #                 mask_value=None, normalized=False),
                # A.Perspective(p=0.01, scale=(0.05, 0.1), keep_size=0, pad_mode=0,
                #                 pad_val=(0, 0, 0), mask_pad_val=0, fit_output=0, 
                #                 interpolation=0),
                # A.PiecewiseAffine(p=0.01, scale=(0.03, 0.05), nb_rows=(4, 4), 
                #                     nb_cols=(4, 4), interpolation=0,
                #                     mask_interpolation=0, cval=0, cval_mask=0,
                #                     mode='constant', absolute_scale=0, 
                #                     keypoints_threshold=0.01),
                # A.RandomCropFromBorders(p=0.01, crop_left=0.1, crop_right=0.1, 
                #                         crop_top=0.1, crop_bottom=0.1),
                # A.CoarseDropout(p=0.01, max_holes=8, max_height=8, max_width=8, 
                #                 min_holes=8, min_height=8, min_width=8, 
                #                 fill_value=(0, 0, 0), mask_fill_value=None),
                # A.PixelDropout(p=0.01, dropout_prob=0.01, per_channel=0, 
                #                 drop_value=(0, 0, 0), mask_drop_value=None),
                # A.RandomFog(p=0.01, fog_coef_lower=0.2, fog_coef_upper=0.2, 
                #             alpha_coef=0.08),
                # A.RandomSnow(p=0.01, snow_point_lower=0.1, snow_point_upper=0.1, 
                #             brightness_coeff=2),
                # A.RandomRain(p=0.01, slant_lower=-10, slant_upper=10, 
                #             drop_length=20, drop_width=1, drop_color=(0, 0, 0),
                #             blur_value=3, brightness_coefficient=0.7, rain_type=None),
                # A.Spatter(p=0.01, mean=(0.65, 0.65), std=(0.3, 0.3), 
                #             gauss_sigma=(2.0, 2.0), intensity=(0.6, 0.6), 
                #             cutout_threshold=(0.68, 0.68), mode=['rain']),
                # A.Spatter(p=0.01, mean=(0.65, 0.65), std=(0.3, 0.3), 
                #             gauss_sigma=(2.0, 2.0), intensity=(0.6, 0.6), 
                #             cutout_threshold=(0.68, 0.68), mode=['mud']),
            ], p=0.8),
    ])
    image_aug = transform(image=image_noaug)['image']
    image_transf = preprocess_input(x = image_aug)
    return image_transf

#Creazioen ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input_aug)

validation_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_dataframe(dataframe = training_data, 
                                                    x_col="Images", 
                                                    y_col="Labels", 
                                                    class_mode="categorical",
                                                    target_size=(224,224),
                                                    batch_size=32)

validation_generator = validation_datagen.flow_from_dataframe(dataframe = val_data, 
                                                              x_col="Images", 
                                                              y_col="Labels", 
                                                              class_mode="categorical",
                                                              target_size=(224,224),
                                                              batch_size=32,
                                                              shuffle=False)

test_generator = test_datagen.flow_from_dataframe(dataframe = test_data, 
                                                              x_col="Images", 
                                                              y_col="Labels", 
                                                              class_mode="categorical",
                                                              target_size=(224,224),
                                                              batch_size=32,
                                                              shuffle=False)



#######################################################################################################



optimizer = SGD()

# Load a pre-trained EfficientDet model
base_model = tf.keras.applications.EfficientNetB0(input_shape=(224,224,3), weights='imagenet', include_top=False)

# Freeze the model layers to prevent them from being updated during training
for layer in base_model.layers:
    layer.trainable = True

# Add a new classification layer on top of the pre-trained model
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x) 
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])

#Addestramento

my_callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=10,
                                mode='auto', restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        '/user/dsilvitelli/kerasenv/bin/Tesi Magistrale/models/urban/best_model.h5',
        monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    keras.callbacks.TensorBoard(log_dir='/user/dsilvitelli/kerasenv/bin/Tesi Magistrale/models/urban/tensorboard'),
    keras.callbacks.CSVLogger(
        '/user/dsilvitelli/kerasenv/bin/Tesi Magistrale/models/urban/epochs_results.csv',
        separator=",", append=False),
]

model.fit(train_generator, epochs=100, validation_data=validation_generator, callbacks=my_callbacks,
            class_weight=weights, use_multiprocessing=True, workers=4)




#########################################       EVALUATE        ############################################
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
from tensorflow import keras
from keras.models import Model
from keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

#Legge test.csv e ricava le colonne interessate
test_data = pd.read_csv("/user/dsilvitelli/kerasenv/bin/Tesi Magistrale/Dataset_finale/Vers_base/CSV/MNV2/Wild/test2.csv", header=0, usecols=[0,1])

test_img_old_path = list(test_data["original_filename_path"])
test_labels = list(test_data["file_class"])

test_img = []
for row in test_img_old_path:
    test_img.append('/user/dsilvitelli/kerasenv/bin' + row[23:])

test_data = pd.DataFrame( {'Images': test_img, 'Labels': test_labels})
test_data.Labels = test_data.Labels.astype(str)

from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_generator = test_datagen.flow_from_dataframe(dataframe = test_data, 
                                                              x_col="Images", 
                                                              y_col="Labels", 
                                                              class_mode="categorical",
                                                              target_size=(224,224),
                                                              batch_size=32,
                                                              shuffle=False)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = load_model('/user/dsilvitelli/kerasenv/bin/Tesi Magistrale/models/wild/ED_wild_pre-trained.h5')

def plot_cm(data, pred):
  y = data["Labels"].values.astype(np.uint8)
  p = pred.argmax(-1)
  cm = confusion_matrix(y, p)
  print('\n\nAccuracy: %.3f' % (y==p).mean())
  print('\n\n', cm)

pred = model.predict(test_generator)
plot_cm(data=test_data, pred=pred)