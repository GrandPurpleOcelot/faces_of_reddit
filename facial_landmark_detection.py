import sys
import os
import dlib
import glob
import cv2

predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = "C:\\Users\\thien\\Google Drive\\Projects and Portfolio\\The face of Reddit\\"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

try:
    for img_file in os.listdir(faces_folder_path):
        if 'jpg' not in img_file:
            pass
        else:

            print("Processing file :{}".format(img_file))
            # open a txt file for each image to record the landmark coordinates
            coordinates_txt = open(faces_folder_path + img_file.split('.')[0] + ".txt","w")

            # read image -> convert to gray scale -> make a copy
            image = cv2.imread(faces_folder_path + img_file)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_copy = image_gray.copy()

            # apply facial detection from Dlib
            faces = detector(image_copy)
            print("Number of faces detected: {}".format(len(faces)))

            # check if a face is presense
            if len(faces) == 0:
                pass
            else:
                # only record the landmark of the first face
                first_face = faces[0]
                print("Left: {} Top: {} Right: {} Bottom: {}".format(first_face.left(), first_face.top(), first_face.right(), first_face.bottom()))
                    
                # Get the landmarks for the face in the box
                landmarks = predictor(image_copy, first_face)
                print("Part 0: {}, Part 1: {} ...".format(landmarks.part(0),
                                                        landmarks.part(1)))

                # Write to txt file            
                for i in landmarks.parts():
                    coordinates_txt.write(str(i.x) + " " + str(i.y) + "\n")

except KeyboardInterrupt:
    print("Keyboard Interrupt")
    sys.exit()