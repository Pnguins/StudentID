import cv2
import numpy as np
import time  # Import the time module

# Start the timer
start_time = time.time()

# Mapping of labels to integers
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


# Function to get the key for a given value in a dictionary
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"


# Function to read and process an image
def read_process_image(file_path):
    # Read the image and resize it
    img = cv2.imread(file_path,1)
    img = cv2.resize(img,(1722,1087), interpolation=cv2.INTER_LINEAR)


    # Define the kernel for sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])


    # Apply the kernel to the image
    img = cv2.filter2D(img, -1, kernel)


    # Extract the region of interest (ROI) for the student ID
    mssv = img[695:750,385:650]
    mssv = cv2.cvtColor(mssv,cv2.COLOR_BGR2GRAY)
   
    # Threshold the image to binary
    th, mssv = cv2.threshold(mssv, 127, 255, cv2.THRESH_BINARY_INV)
   
    # Define the kernel for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    mssv = cv2.erode(mssv, kernel,iterations=1)
    return img, mssv


# Function to find contours and detect digits
def find_contour_detect(img, mssv):
    # Find contours in the image
    contours, _ = cv2.findContours(mssv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Filter out small contours
    min_contour_area = 150  # You may need to adjust this value
    contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]


    # Get the bounding rectangles for each contour
    bounding_rects = [cv2.boundingRect(contour) for contour in contours]


    # Sort the bounding rectangles from left to right
    sorted_rects = sorted(bounding_rects, key=lambda x: x[0])
    studentID = ""
    for rect in sorted_rects:
        # Get the coordinates of the rectangle
        x, y, w, h = rect
        roi = mssv[y:y+h, x:x+w]
        studentID+= digit_detect(roi)


        # Draw rectangle around each digit on the original image
        cv2.rectangle(img, (x+385, y+695), (x+w+385, y+h+695), (0, 255, 0), 2)


    # Return the detected student ID
    return studentID


# Function to find contours in a given region of the image
def find_contour(img, numbers,offset_x,offset_y):
    numbers= cv2.cvtColor(numbers,cv2.COLOR_BGR2GRAY)
    th, numbers = cv2.threshold(numbers, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(numbers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 100
    contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x+offset_x, y+offset_y), (x+w+offset_x, y+h+offset_y), (0, 255, 0), 2)


# Function to detect a digit in a given region of interest
def digit_detect(roi):
    roi = cv2.resize(roi, (30,30), interpolation = cv2.INTER_LINEAR)


    # Load the trained SVM model
    svm = cv2.ml.SVM_load('svm_data.dat')


    # Define the parameters for the HOG descriptor
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


    # Compute the HOG descriptor for the ROI
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)


    hog_features = hog.compute(roi)
    hog_features = np.float32(hog_features).reshape(-1, hog_features.shape[0])
   
    # Predict the digit using the SVM model
    prediction = svm.predict(hog_features)
    return get_key(labels_map,prediction[1][0])


# Read and process the image
img, mssv = read_process_image('the_sv/img_04.jpg')


# Find contours in different regions of the image
find_contour(img,img[380:473,490:1179],490,380)
find_contour(img,img[475:530,800:920],800,475)
find_contour(img,img[475:530,1050:1180],1050,475)
find_contour(img,img[630:700,370:720],370,630)
find_contour(img,img[685:760,915:1225],915,685)
find_contour(img,img[815:890,1055:1370],1055,815)


# Detect the student ID
studentID = find_contour_detect(img, mssv)


# Resize the image and display the student ID
img = cv2.resize(img,(1024,720),interpolation=cv2.INTER_LINEAR)
print(f"Student ID: {studentID}")


# Display the image
cv2.imshow("img", img)
cv2.waitKey(0)
# Stop the timer
end_time = time.time()

# Calculate the running time
running_time = end_time - start_time

print(f"Running time: {running_time} seconds")