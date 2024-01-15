import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

labels_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8, 
    "9": 9,
    "h": 10,
    "k": 11,
    "c": 12,
}

images = []
labels = []

folder_path = 'output_images'

# Get a list of all filenames in the folder
filenames = os.listdir(folder_path)

# Filter the list for filenames ending with .jpg, .jpeg, .png, etc.
image_filenames = [f for f in filenames if f.endswith(('.png', '.jpg', '.jpeg'))]

datagen = ImageDataGenerator(
    rotation_range=1,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    shear_range=0.1,  # randomly apply shearing transformations
    zoom_range=0.1,  # randomly zoom in and out
)


for image_filename in image_filenames:
    img = cv2.imread(os.path.join(folder_path,image_filename),cv2.IMREAD_GRAYSCALE)
    th, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    if img is not None:
        img = img.reshape((1,) + img.shape + (1,))  # reshape image to (1, height, width, 1) for data augmentation
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_image = batch[0].astype('uint8')
            images.append(augmented_image)
            labels.append(labels_map[image_filename.split("_")[-1].split('.')[0]])
            i += 1
            if i > 800:  # save 800 augmented images
                break

images = np.array(images)
labels = np.array(labels)

print(labels)

winSize = (30,30)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (5,5)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradient = True

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

hog_descriptors = []
for img in images:
    hog_descriptors.append(hog.compute(img))
hog_descriptors = np.squeeze(hog_descriptors)

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(hog_descriptors, cv2.ml.ROW_SAMPLE, labels)
svm.save('svm_data.dat')
