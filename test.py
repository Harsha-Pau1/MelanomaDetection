import os
import scipy.io
import numpy as np
from glob import glob
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from skimage.transform import resize
dir_path = 'C:/Users/91807/Documents/HARSHA/MainProject/HSI_datacube_after_correction'
sub_dir = glob(dir_path+"/*")
#print(sub_dir)
MMCube_list=list(glob(dir_path+'/MMCube/*'))
DNCube_list=list(glob(dir_path+'/DNCube/*'))
OtherCube_list=list(glob(dir_path+'/OtherCube/*'))
print("length of mmcube:",len(MMCube_list))
print("length of dncube:",len(DNCube_list))
print("length of other cube:",len(OtherCube_list))
IEC=[]
Lentigo=[]
Melt=[]
Nv=[]
Sebk=[]
#OtherCube = os.listdir('/content/drive/MyDrive/HSI_datacube_after_correction/OtherCube')
for path in OtherCube_list:
  if 'IEC' in path:
    IEC.append(path)
  elif 'Lentigo' in path:
    Lentigo.append(path)
  elif 'Melt' in path:
    Melt.append(path)
  elif 'Nv' in path:
    Nv.append(path)
  else:
    Sebk.append(path)
input_map={'MMCube':MMCube_list,'DNCube':DNCube_list,'IEC':IEC,'Lentigo':Lentigo,'Melt':Melt,'Nv':Nv,'Sebk':Sebk}
output_map={'MMCube':0,'DNCube':1,'IEC':2,'Lentigo':3,'Melt':4,'Nv':5,'Sebk':6}
x = loadmat('C:/Users/91807/Documents/HARSHA/MainProject/HSI_datacube_after_correction/OtherCube/Lentigo16.mat')
print(x.keys())
# 'DataCubeC' is the image array
# Extract the image array from the MATLAB file
img_array = x['DataCubeC']
# Shape of the image array
img_array.shape
# 272: number of pixels in the spatial (x-y) dimension of the image along the x-axis.
# 512: number of pixels in the spatial (x-y) dimension of the image along the y-axis.
# 16: number of spectral bands (or wavelengths) in the spectral dimension of the image.
x_list=[]
y_list=[]
img_arr=[]
for disease in input_map:
  path_list=input_map[disease]
  for mat in path_list:
    # Load the MATLAB file
    mat_file=loadmat(mat)
    # Extract the image array from the MATLAB file
    image_array = mat_file['DataCubeC']
    # Normalize the pixel values
    max_value = np.max(image_array)
    image_data_normalized = image_array / max_value
    resized_img = resize(image_data_normalized, (224, 224))
    x_list.append(resized_img)
    y_list.append(output_map[disease])
  print("Extracting... :",disease)
print("Completed!")
X = np.array(x_list)
y = np.array(y_list)
X.shape
# Apply data augmentation
# Randomly flip images horizontally and vertically
X = tf.image.random_flip_left_right(X)
X = tf.image.random_flip_up_down(X)
X.ndim
#Split the data into training, validation, and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)