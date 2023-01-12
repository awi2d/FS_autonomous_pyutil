#copied from https://github.com/ayush111111/cone_keypoint_regression/blob/master/keypoint_regression80.ipynb
#and slightly changed.

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import csv
import onnx
import tf2onnx
from util import getType

dataset_dir = "C:/Users/Idefix/PycharmProjects/datasets/keypoints/"
labels_file = dataset_dir+"annotations_tmp.csv"
targetSize = (80,80)


def get_scale(orig_im_size ,targetSize):
    target_h, target_w = targetSize
    h_scale = target_h / orig_im_size[0]
    w_scale = target_w / orig_im_size[1]
    return h_scale, w_scale


def plot_keypoints(img, points):
    plt.imshow(img, cmap='gray')
    for i in range(0,14,2):
        plt.scatter((points[i] + 0.5)*targetSize[0], (points[i+1]+0.5)*targetSize[0], color='red')


#flip along vertical axis
def lrFlip(img, points):

    new_img = np.copy(img)
    new_img = np.fliplr(new_img)

    new_points = np.copy(points)
    for i in range(0,len(new_points),2):
        new_points[i] = -points[i]

    return new_img, new_points




def adj_intensity(img):

    new_img = np.copy(img)
    inc_brightness_images = np.clip(new_img*1.2, 0.0, 1.0)
    dec_brightness_images = np.clip(new_img*0.8, 0.0, 1.0)

    return inc_brightness_images, dec_brightness_images


def read_data():
    with open(labels_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader)
        y = []
        for row in reader:
            y.append(row)

    y_dat = pd.DataFrame(y)

    # read images
    inputImages = []
    for filename in [dataset_dir+fn for fn in y_dat[0]]:
        img = cv2.imread(filename)
        #imread as grayscale

        img = cv2.resize(img, targetSize)
        inputImages.append(img)#.reshape(targetSize[0],targetSize[1],1))

    cv2.imshow("img", inputImages[1])
    #plt.imshow(inputImages[1])

    images = np.array(inputImages, dtype='float32')

    # process Labels
    y_dat = pd.DataFrame(y_dat.values[:], columns=header)
    print("y_dat.columns = ", y_dat.columns)
    y_clipped = y_dat[y_dat.columns[2:]]  # y_dat.columns[2:-1] = ndex(['top', 'mid_L_top', 'mid_R_top', 'mid_L_bot', 'mid_R_bot', 'bot_L', 'bot_R], dtype='object')
    #y_clipped = labels without filenames and URI
    #y_dat.drop(y_dat.index[0])

    inpKeypointsX = pd.DataFrame()
    inpKeypointsY = pd.DataFrame()
    for column in y_clipped.columns:
        temp = y_clipped[column].str.strip('[]').str.split(",", expand=True)
        inpKeypointsX[column+'_x'] = temp[0]
        inpKeypointsY[column+'_y'] = temp[1]
    #print("inpKeypointsX", inpKeypointsX)  # subset of y_clipped, where only the x_variables exist

    inpKeypointsX_arr = np.array(inpKeypointsX,dtype= 'float32')
    inpKeypointsY_arr = np.array(inpKeypointsY,dtype= 'float32')

    #Scale Images
    KeyptsXscaled = pd.DataFrame()
    KeyptsYscaled = pd.DataFrame()

    for i in range(len(inputImages)):
        KeyptsXscaled[i] = inpKeypointsX_arr[i]*targetSize[0]  # added*image_size to change from (relative to img size) to pixel position
        KeyptsYscaled[i] = inpKeypointsY_arr[i]*targetSize[1]
    KeyptsXscaled = KeyptsXscaled.copy()
    KeyptsYscaled = KeyptsYscaled.copy()
    KeyptsXscaled = KeyptsXscaled.T
    KeyptsYscaled = KeyptsYscaled.T
    KeyptsX = pd.DataFrame(data=KeyptsXscaled.values, columns=inpKeypointsX.columns)
    KeyptsY = pd.DataFrame(data=KeyptsYscaled.values, columns=inpKeypointsY.columns)

    keypt = pd.DataFrame()
    for column in range(len(KeyptsY.columns)):
        keypt[KeyptsX.columns[column]]=KeyptsX.iloc[:,column]
        keypt[KeyptsY.columns[column]]=KeyptsY.iloc[:,column]

    keypt = np.ceil(keypt)

    # bring input and output into range [-0.5, 0.5] and [0, 1].
    keypt_arr = np.array(keypt,dtype= 'float32')
    keypt_arr = keypt_arr/targetSize[0] - 0.5

    images = images/255

    print("labels: ", getType(keypt_arr))
    print("images/input: ", getType(images))

    # show images with keypoints
    #fig = plt.figure(figsize=(10, 10))
    #for id in range(20):
    #    fig.add_subplot(4, 5, id + 1, xticks=[], yticks=[])
    #    plot_keypoints(inputImages[id], keypt_arr[id])
    #plt.show()

    #Augmentation
    flipped_images = []
    flipped_keypts = []
    inc_intensity_images = []
    inc_intensity_keypts = []
    dec_intensity_images = []
    dec_intensity_keypts = []
    rotated_images = []
    rotated_keypts = []

    for i,image in enumerate(images):
        flip_img, flip_points = lrFlip(images[i], keypt_arr[i])
        flipped_images.append(flip_img)
        flipped_keypts.append(flip_points)

        inc_intensity_image, dec_intensity_image = adj_intensity(images[i])
        inc_intensity_images.append(inc_intensity_image)
        inc_intensity_keypts.append(keypt_arr[i])
        dec_intensity_images.append(dec_intensity_image)
        dec_intensity_keypts.append(keypt_arr[i])


    angle = 15
    #15 degrees
    center = (targetSize[0]/2, targetSize[1]/2)
    for angle in [-angle, angle]:

        RotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
        for image in images:
            rotated_image = cv2.warpAffine(image, RotMat, targetSize, flags=cv2.INTER_CUBIC)
            rotated_images.append(rotated_image)#.reshape(targetSize[0],targetSize[1],1))

        angle_rad = -angle*np.pi/180.
        for keypt in keypt_arr:
            rotated_keypt = (keypt)*targetSize[0]
            for x in range(0,len(rotated_keypt), 2):
                rotated_keypt[x] = rotated_keypt[x]*np.cos(angle_rad) - rotated_keypt[x+1]*np.sin(angle_rad)
                #x+1 is y
                rotated_keypt[x+1] = rotated_keypt[x]*np.sin(angle_rad) + rotated_keypt[x+1]*np.cos(angle_rad)

            rotated_keypt = (rotated_keypt)/targetSize[0]
            rotated_keypts.append(rotated_keypt)




    flipped_images_arr = np.array(flipped_images,dtype= 'float32')
    flipped_keypts_arr = np.array(flipped_keypts,dtype= 'float32')
    inc_intensity_images_arr = np.array(inc_intensity_images,dtype= 'float32')
    inc_intensity_keypts_arr = np.array(inc_intensity_keypts,dtype= 'float32')
    dec_intensity_images_arr = np.array(dec_intensity_images,dtype= 'float32')
    dec_intensity_keypts_arr = np.array(dec_intensity_keypts,dtype= 'float32')
    rotated_images_arr = np.array(rotated_images,dtype= 'float32')
    rotated_keypts_arr = np.array(rotated_keypts,dtype= 'float32')

    #fig = plt.figure(figsize=(10, 10))
    #fig.add_subplot(4, 5, 2, xticks=[], yticks=[])
    #plot_keypoints(flipped_images[1], flipped_keypts[1])
    #fig.add_subplot(4, 5, 3, xticks=[], yticks=[])
    #plot_keypoints(inputImages[1], keypt_arr[1])
    #fig.add_subplot(4, 5, 4, xticks=[], yticks=[])
    #plot_keypoints(inc_intensity_images[1], inc_intensity_keypts[1])
    #fig.add_subplot(4, 5, 5, xticks=[], yticks=[])
    #plot_keypoints(dec_intensity_images[1], dec_intensity_keypts[1])
    #fig.add_subplot(4, 5, 6, xticks=[], yticks=[])
    #plot_keypoints(rotated_images[1], rotated_keypts[1])
    #plt.show()

    print(f"fliped: {getType(flipped_images_arr)}, {getType(flipped_keypts_arr)}\ninc_intesity: {getType(inc_intensity_images_arr)}, {getType(inc_intensity_keypts_arr)}")
    print(f"dec_intesity: {getType(dec_intensity_images_arr)}, {getType(dec_intensity_keypts_arr)}\nrotated: {getType(rotated_images_arr)}, {getType(rotated_keypts_arr)}")

    keypts_final = np.concatenate((keypt_arr, flipped_keypts_arr))
    keypts_final = np.concatenate((keypts_final, rotated_keypts_arr))
    keypts_final = np.concatenate((keypts_final, inc_intensity_keypts_arr))
    keypts_final = np.concatenate((keypts_final, dec_intensity_keypts_arr))

    images_final =  np.concatenate((images, flipped_images_arr))
    images_final =  np.concatenate((images_final, rotated_images_arr))
    images_final =  np.concatenate((images_final, inc_intensity_images_arr))
    images_final =  np.concatenate((images_final, dec_intensity_images_arr))


    return images_final, keypts_final

def ResNetBlock(input_data, C):

    x = tf.keras.layers.Conv2D(filters=C,kernel_size=(3,3),padding='same',dilation_rate=(2,2))(input_data)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #LeakyReLU(alpha = 0.1)

    x = tf.keras.layers.Conv2D(filters=C,kernel_size=(3,3),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = Activation('relu')(x)
    #x = Dropout(0.2)(x)

    x_shortcut = tf.keras.layers.Conv2D(filters=C,kernel_size=(1,1),padding='same')(input_data)
    x_shortcut = tf.keras.layers.BatchNormalization()(x_shortcut)

    x = tf.keras.layers.Add()([x,x_shortcut])
    #x = Dropout(0.3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    #LeakyReLU(alpha = 0.1)
    #Dropout()?
    return x


def get_Model():
    filters = 64
    len_classes = 14
    inputs = tf.keras.Input(shape = (targetSize[0],targetSize[0],3))

    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=7,strides=(1,1),padding='same',kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #LeakyReLU(alpha = 0.1)
    #x = Dropout(0.2)(x)
    x = ResNetBlock(x,16)
    x = ResNetBlock(x,32)
    x = ResNetBlock(x,64)
    #x = ResNetBlock(x,128)


    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(len_classes)(x)
    CRmodel = tf.keras.models.Model(inputs, outputs)

    #Same convolutions using a 3×3 kernel via a residual block
    #C= 64,C= 128,C= 256 and C= 512 for dhal et al 2019 , filters = 64
    return CRmodel


def delta2D(pointA_x,pointA_y,pointB_x,pointB_y):
    return tf.sqrt(tf.square(pointA_y-pointB_y)+tf.square(pointA_x-pointB_x))


def cross_ratio(point1_x,point1_y,point2_x,point2_y,point3_x,point3_y,point4_x,point4_y):
    num = delta2D(point1_x,point1_y,point3_x,point3_y)/delta2D(point2_x,point2_y,point3_x,point3_y)
    den = delta2D(point2_x,point2_y,point3_x,point3_y)/delta2D(point2_x,point2_y,point4_x,point4_y)
    return num/den


""" 
  arguments y_true,y_pred values of dim (batch_size,14)
  output tensor of (batch_size,) with cross ratio of individual rows 
  
"""
def TrialCrossRatioLoss(Y_true, Y_pred):
    gamma = 0.0001
    cr3D = 1.39408

    cr_left_arm = cross_ratio(Y_true[:,0],Y_true[:,1],Y_true[:,2],Y_true[:,3],Y_true[:,6],Y_true[:,7],Y_true[:,10],Y_true[:,11])
    cr_right_arm = cross_ratio(Y_true[:,0],Y_true[:,1],Y_true[:,4],Y_true[:,5],Y_true[:,8],Y_true[:,9],Y_true[:,12],Y_true[:,13])

    sum_points_x = tf.square(Y_true[:,0]-Y_pred[:,0]) + tf.square(Y_true[:,2]-Y_pred[:,2]) + tf.square(Y_true[:,6]-Y_pred[:,6]) +tf.square(Y_true[:,10]-Y_pred[:,10]) +tf.square(Y_true[:,4]-Y_pred[:,4]) +tf.square(Y_true[:,8]-Y_pred[:,8]) +tf.square(Y_true[:,12]-Y_pred[:,12])
    sum_points_y = tf.square(Y_true[:,1]-Y_pred[:,1]) + tf.square(Y_true[:,3]-Y_pred[:,3]) + tf.square(Y_true[:,7]-Y_pred[:,7]) +tf.square(Y_true[:,11]-Y_pred[:,11]) +tf.square(Y_true[:,5]-Y_pred[:,5]) +tf.square(Y_true[:,9]-Y_pred[:,9]) +tf.square(Y_true[:,13]-Y_pred[:,13])

    cross_ratio_sum = gamma*tf.square(cr_left_arm-cr3D) + gamma*tf.square(cr_right_arm-cr3D)

    total_sum = sum_points_x + sum_points_y + cross_ratio_sum

    return total_sum


def train(model, images, keypts):
    X_train, X_test, y_train, y_test = train_test_split(images, keypts, test_size=0.2, random_state=42)
    lr = 0.001
    batch_size = 128
    optimizer = tf.keras.optimizers.Adam(lr)
    metrics = tf.keras.metrics.MeanSquaredError()
    save_dir = "training_1/"

    model.compile(loss=TrialCrossRatioLoss, optimizer=optimizer, metrics=[metrics])

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir+"checkpoint_best.ckpt", save_weights_only=True, verbose=1, save_best_only=True)
    hist = model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), batch_size = batch_size, callbacks=[checkpoint], shuffle=True, verbose=1)
    print("hist = \n", hist)
    model.save(save_dir+"trained_keypoint_regression_model.h5")


def predict(model, img):
    if len(img.shape) == 3: #single image
        return model.predict(np.array([img]))[0]
    elif len(img.shape) == 4: # list of images
        return model.predict(np.array(img))


def main():
    #train
    images, keypts = read_data()
    model = get_Model()
    model.load_weights("training_1/trained_keypoint_regression_model.h5") # optionally load pretrained weights
    #model.summary()

    #TODO train with more & better images
    #train(model, images, keypts)

    fig = plt.figure(figsize=(10, 10))
    predictions = predict(model, images[:25])
    for id in range(25):
        fig.add_subplot(5, 5, id + 1, xticks=[], yticks=[])
        plot_keypoints(images[id], predictions[id])
    plt.show()  # model kinda works, but points are way to scatterd
    input_signature = [tf.TensorSpec([None, targetSize[0], targetSize[1], 3], tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save_model(onnx_model, "keypoint_regression_best.onnx")

