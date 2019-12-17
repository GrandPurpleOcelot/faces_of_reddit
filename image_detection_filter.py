import numpy as np
import cv2
import os
import shutil

PATH_TO_DATA_LOCATION_WINDOWS = "E:\img_data"
PATH_TO_SELECTED_IMAGES_FOLDER_WINDOWS = "E:\selected_images"

# import haar cascade facial detection model.
# source: https://github.com/parulnith/Face-Detection-in-Python-using-OpenCV/tree/master/data/haarcascades

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get list of image file names: 
file_list = os.listdir(PATH_TO_DATA_LOCATION_WINDOWS)
image_list = []
for file in file_list:
    if file.split(".")[1] in ['jpg', 'jpeg']:
        image_list.append(file)

def image_qualifier(image, image_classifier = haar_cascade_face, minNeighbors = 20):
    try:
        image_read = cv2.imread(PATH_TO_DATA_LOCATION_WINDOWS + "\\" + image)
        image_copy = image_read.copy()

        # reduce the resolution of the image by 50% with image excess 1440p:
        if image_copy.shape[0] >= 1440 or image_copy.shape[1] >= 1440:
            image_copy = cv2.resize(image_copy,None,fx=0.2,fy=0.2)
    
        gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        faces_rects = image_classifier.detectMultiScale(gray_image, minNeighbors= minNeighbors)

    except AttributeError:
        print("{} returns Attribute Errors".format(image))
        return 0
    
    return(len(faces_rects))

# keep count of qualified images for each sub
subreddit_count = dict.fromkeys([image.split("_")[0] for image in image_list],0)

# list of qualified images
qualify_images = []

for image_name in image_list:
    if subreddit_count[image_name.split("_")[0]] >= 50:
        pass
    else:
        if image_qualifier(image_name) > 0:
            qualify_images.append(image_name)
            subreddit_count[image_name.split("_")[0]] += 1

# Move qualified images to selected_images folder:
for image in qualify_images:
    shutil.move(PATH_TO_DATA_LOCATION_WINDOWS + "\\" + image, PATH_TO_SELECTED_IMAGES_FOLDER_WINDOWS + "\\" + image)

print([(k,v) for k,v in subreddit_count.items()])
        
