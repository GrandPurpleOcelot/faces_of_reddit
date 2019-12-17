# The Faces of Reddit: Average Face

## Overview
The project is set out to construct the average face for each subreddit using the images harvested from the user submissions from a number of given subreddits. The key analysis includes a single face generated from the aggregation of multiple faces detected in the images. The data science methods include computer vision (facial recognition) and Delaunay triangulation of facial landmarks.



## Motivation

As social primates, humans always seem to be particularly interested in the facial features of one another. In the paper published in 1990, psychologist Judith Langlois and Lori Roggman found out that attractive faces are only average (Langlois & Roggman).

In particular, we tend to perceive the faces manipulated to make their shapes closer to the average as more attractive (Valentine et al, 2004). Without diving too much into human bias and social science, I would like to use data science as a tool to explore many different techniques involve human facial detection and learn how to generate average faces from any given community. 

Inspired by a submission from Vincent Riemer, a Redditor from the subreddit Dataisbeautiful: “The Average Faces of 42 Different Subreddits”, I would like to propose to explore a project with a similar approach to generate average faces for a small number of subreddits. With the knowledge foundation established in this project, I hope to develop an open-source tool that everybody can use to generate an average face from input images.


## Outline of the Project

* Data scope and collection

* Data pre-processing

* Delauney Triangulation

* Face Averaging

Each section will contain a detail description and discussion of employ methods. Methods from this project were adapted from Satya Mallick's [OpenCV tutorial series](https://www.learnopencv.com/average-face-opencv-c-python-tutorial/). 


```python
# import required library
import os
import cv2
import numpy as np
import math
import sys
import dlib
import matplotlib.pyplot as plt
import praw
import requests
```

## Data Scope and Collection

### Data Source

The image data for this project include images submission sorted by "Top" from a number of pre-selected subreddits. 

**Retrieval Method:**

1. Image URLs can be obtained from [Reddit API](https://www.reddit.com/dev/api/) accessible via [Python Reddit API Wrapper](https://praw.readthedocs.io/en/latest/)

2. Image data is download from collected URLs using the Python library: requests

In this project, 50 qualify image submissions that contain at least 1 human face will be used for face averaging. For demonstration purposes, we will perform face averaging using 4 sample images from subreddit r/headshots.


```python
IMAGE_FORMAT = ['jpeg', 'jpg']
SUBREDDIT_LIST = ['headshots']

img_list = {}

def get_img_url(subreddit_name, limit = 5):
    reddit = praw.Reddit(client_id='ENTER_REDDIT_APIT_CREDENTIAL',
                     client_secret='ENTER_REDDIT_APIT_CREDENTIAL',
                     user_agent='image_crawler_v0.0')

    try:
        count = 0
        submissions = reddit.subreddit(subreddit_name).top(limit = limit)

        for submission in submissions:
            file_name = subreddit_name + '_' + submission.url.split('/')[-1]
            if file_name.split('.')[-1] not in IMAGE_FORMAT:
                continue 
            else:
                img_list[file_name] = submission.url
                count +=1
        
        print("{} query  - {} results".format(subreddit_name, count))

    except Exception:
        print("Subreddit r/{} does not exist".format(subreddit_name))
        pass

def download_img(subreddit_list):
    for subreddit in subreddit_list:
        get_img_url(subreddit)
    
    error_count = 0
    for key, value in img_list.items():
        try:
            r = requests.get(value)
        except Exception:
            error_count += 1
            pass
        else:
            file_path = os.path.join(os.getcwd(), key)
            with open(file_path, mode = 'wb') as f:
                f.write(r.content)
    
    print("Number of dead URLs: {}".format(error_count))
    
    
download_img(SUBREDDIT_LIST)
```

    headshots query  - 4 results
    Number of dead URLs: 0


Let's take a quick look at these 4 images that contain frontal human faces:


```python
images = []

for filename in sorted(os.listdir(os.getcwd())):
    if filename.endswith(".jpg"):
        images.append(filename)

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(40,100))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(convertToRGB(cv2.imread(image)))
```


![png](Final%20Report/output_7_0.png)


# Data Pre-processing

Since the data is in image format, there isn't much to do in terms of data cleaning. However, it is important to process these images in a standard format that the algorithm can consume. There are a number of steps that we need to do first:

1. Convert to gray-scale

2. Detect face and construct facial landmarks

3. Image Standardization

## 1. Convert to gray-scale

A typical image is represented as a combination of Red, Blue, and Green, and all other colors can be achieved by the combination of these three basic colors. However, the popular facial detection implementations such as those found in OpenCV or Dlib expect gray-scaled images. There is a good reason for doing this. The gray image only contains a single channel instead of three in RGB, thus, reducing the memory and processing resources that the computer has to handle.



```python
test_image = cv2.imread(images[0])
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
plt.imshow(test_image_gray, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x27b3cf18c50>




![png](Final%20Report/output_9_1.png)


## 2. Detect face and construct facial landmarks

We will use the human face detection algorithm implemented in [Dlib](http://dlib.net/) - a multipurpose toolkit containing multiple useful algorithms, including computer vision applications. This method employs the regression trees classification algorithm outlined in the Kazemi and Sullivan’s paper (2014). The output is a set of coordinates that denote the position of the rectangular that contain a human face. We need two points to draw a rectangular: upper-left point and lower right point. 

### Facial Detection

We first need to filter out the images that do not contain at least a human face.


```python
detector = dlib.get_frontal_face_detector()
faces = detector(test_image_gray)

image_copy = test_image_gray.copy()

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(img = image_copy, pt1 = (x1, y1), pt2 = (x2, y2), color = (255, 0, 0), thickness = 10)
    
    plt.imshow(image_copy, cmap='gray')
```


![png](Final%20Report/output_11_0.png)


We can see the algorithm successfully detected a human face in this image, denoted by the white rectangular that enclose the outer limits of facial features.

### Facial landmarks construction

Human facial landmarks can be constructed within the detected rectangular. Dlib’s facial landmark predictor was trained on the iBUG 300-W dataset (Sagonas et al., 2016). The output contains x and y coordinates of 68 facial landmarks for any given face.


```python
predictor = dlib.shape_predictor("C:\\Users\\thien\\Google Drive\\Projects and Portfolio\\The face of Reddit\\shape_predictor_68_face_landmarks.dat")

landmarks = predictor(test_image_gray, face)

for n in range(0, 68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    cv2.circle(image_copy, (x, y), 15, (255, 0, 0), -1)
    
    plt.imshow(image_copy, cmap='gray')
```


![png](Final%20Report/output_14_0.png)


This is an important method that I will use to construct 68 landmarks on the human face. These points denote the physical features of a face including the location of the eyes, nose, lips, and jawline. The points will be used to construct Delaunay triangulation for facial morphing in the next step of the face averaging process.

I provided a script that can detect and store the facial landmarks to a text file that we can retrieve for later uses. 


```python
predictor_path = "C:\\Users\\thien\\Google Drive\\Projects and Portfolio\\The face of Reddit\\shape_predictor_68_face_landmarks.dat"
faces_folder_path = os.getcwd() + "\\"

try:
    for img_file in os.listdir(faces_folder_path):
        if not img_file.endswith(".jpg"):
            pass
        else:
            # open a txt file for each image to record the landmark coordinates
            coordinates_txt = open(faces_folder_path + img_file.split('.')[0] + ".txt","w")

            # read image -> convert to gray scale -> make a copy
            image = cv2.imread(faces_folder_path + img_file)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_copy = image_gray.copy()

            # apply facial detection from Dlib
            faces = detector(image_copy)

            # check if a face is presense
            if len(faces) == 0:
                pass
            else:
                # only record the landmark of the first face
                first_face = faces[0]
                    
                # Get the landmarks for the face in the box
                landmarks = predictor(image_copy, first_face)

                # Write to txt file            
                for i in landmarks.parts():
                    coordinates_txt.write(str(i.x) + " " + str(i.y) + "\n")

except KeyboardInterrupt:
    print("Keyboard Interrupt")
    sys.exit()
```

## 3. Image Standardization

We have generated 68 facial landmarks for each image. However, facial landmarks are different for each facial image. Therefore, we need to bring the faces into the same reference frame. To do this, we can use the similarity transform that will transform the points from the input image coordinates to the standard output image coordinates.

Let's retrive the coordinates that we store for the sample image:


```python
#Create an array of points.
points = []
            
# Read points from the .txt file:
with open(images[0].split('.')[0] + ".txt") as file :
    for line in file :
        x, y = line.split()
        points.append((int(x), int(y)))
            
print(points)
```

    [(1234, 439), (1225, 531), (1223, 623), (1230, 712), (1261, 791), (1323, 858), (1396, 911), (1474, 955), (1550, 974), (1614, 960), (1663, 908), (1711, 856), (1754, 795), (1782, 729), (1810, 660), (1831, 594), (1839, 524), (1379, 385), (1437, 362), (1500, 363), (1559, 380), (1609, 412), (1710, 429), (1751, 415), (1794, 411), (1835, 417), (1856, 448), (1651, 481), (1651, 544), (1653, 607), (1655, 670), (1568, 692), (1597, 704), (1625, 716), (1652, 713), (1675, 704), (1441, 453), (1481, 439), (1524, 445), (1555, 478), (1514, 481), (1473, 475), (1696, 498), (1734, 474), (1772, 478), (1796, 500), (1771, 514), (1732, 510), (1466, 767), (1534, 774), (1588, 770), (1615, 783), (1641, 775), (1662, 784), (1677, 795), (1649, 822), (1624, 839), (1595, 842), (1564, 835), (1518, 812), (1484, 775), (1582, 794), (1610, 800), (1637, 797), (1661, 795), (1630, 796), (1603, 800), (1574, 793)]


Each point is represented as a tuple of (x,y) coordinates. There should be 68 tuples that represent 68 facial landmarks.

For image croping, we only need two points from the corner of the eyes, which correspond to index 36 and 46 in the points array:


```python
image_copy = test_image.copy()

for (x,y) in [points[index] for index in [36,45]]:
    cv2.circle(image_copy, (x, y), 20, (0, 0, 255), -1)

plt.imshow(convertToRGB(image_copy), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x27b3d0fb1d0>




![png](Final%20Report/output_20_1.png)


To standardize the input images, we want the face to align horizontally by using certain points on the image. For example, we can use a similarity transform to rotate the position of the points of two eyes to the standard position (horizontal). We will specify the output with a pixel density of 600x600.


```python
# codes adapted from https://github.com/spmallick/learnopencv/tree/master/FaceAverage

image_copy = test_image.copy()

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.

def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)  
  
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
    
    inPts.append([np.int(xin), np.int(yin)])

    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
    
    outPts.append([np.int(xout), np.int(yout)])
    
    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    
    return tform[0]

# Dimensions of output image
w = 600
h = 600

eyecornerSrc  = [points[36], points[45] ] 
eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ]


# Compute similarity transform
tform = similarityTransform(eyecornerSrc, eyecornerDst)

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

img = cv2.warpAffine(image_copy, tform, (w,h))
plt.imshow(convertToRGB(img))
```




    <matplotlib.image.AxesImage at 0x27b3d1800f0>




![png](Final%20Report/output_22_1.png)


We can see the face has been rotated horizontally with respect to eye line and cropped with a size of 600 x 600. This standardized face will be much easier to morph, blend, and average with other images.

# Naive Face Averaging


Now that the images are of the same size, we can obtain the average image by averaging the pixels values of all images, right? Not quite. The main problem is the misalignment of each individual face. Since we only aligned the corners of two eyes, the other facial features may not align properly.

Here's a demonstration of this naive approach:


```python
aggregate = np.zeros((h,w,3), np.float32())

for image in transformed_images:
    img = cv2.imread(image)
    aggregate = aggregate + img

naive_average_image = aggregate/3

plt.imshow(convertToRGB(naive_average_image.astype(np.uint8)))
```




    <matplotlib.image.AxesImage at 0x27b3ce604e0>




![png](Final%20Report/output_25_1.png)


We can see that the result is an abomination that results from facial features mismatching. To fix this issue, we can morph the face into an average triangles mesh using Delaunay Triangulation.   

## Delaunay Triangulation

Since we don't have the information to perfectly match the set of points from one image to another, we can take advantage of Delaunay Triangulation to generate a set of triangles that can be aligned. Delaunay triangulation allows us to break the image down into a smaller set of triangles that correspond to the 68 points.

Fortunately, the cv2 library already provides us with appropriate tools for calculating the Delauney Triangulation based on given points. We need to make sure to transform the 68 landmarks using similarity transform as well:


```python
# Apply similarity transform on 68 landmarks
# points need to be in array format before apply transformation
points1 = np.reshape(np.array(points), (68,1,2))
points1_transformed = cv2.transform(points1, tform)
points_copy = np.reshape(points1_transformed, (68,2))

# return points to list format
transformed_coordinates = points_copy.tolist()

# calculate the 3rd order polygon for 68 points
convexhull = cv2.convexHull(points_copy)

# Delaunay triangulation
rect = cv2.boundingRect(convexhull)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(transformed_coordinates)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype = np.int32)

print(triangles[0])
```

    [180 200  42 209 133 160]


Each triangle need at least three point, which represented by six numbers for x and y coordinates of said points. Let's visualize the triangle mesh:


```python
for t in triangles:
    point1 = (t[0], t[1])
    point2 = (t[2], t[3])
    point3 = (t[4], t[5])

    # draw lines that connect point 1, 2, and 3 of all triangles
    cv2.line(img, point1, point2, (0,0,255), 5)
    cv2.line(img, point1, point3, (0,0,255), 5)
    cv2.line(img, point2, point3, (0,0,255), 5)
    
plt.imshow(convertToRGB(img))
```




    <matplotlib.image.AxesImage at 0x27b3d0ab6a0>




![png](Final%20Report/output_30_1.png)


Let's create a function for the steps above. The outputs are the cropped images and coordinates of the transformed landmarks.  


```python
def image_processing(image_name):
    try:
        print("Processing {}...".format(image_name))
        test_image = cv2.imread(image_name)

        #Create an array of points.
        points = []

        # Read points from the .txt file:
        with open(image_name.split('.')[0] + ".txt") as file :
            for line in file :
                x, y = line.split()
                points.append((int(x), int(y)))

        image_copy = test_image.copy()

        eyecornerSrc  = [points[36], points[45] ] 
        eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ]

        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        transformed_img = cv2.warpAffine(image_copy, tform, (w,h))

        cv2.imwrite(os.getcwd() + "\\transformed_" + image_name, transformed_img)

        points1 = np.reshape(np.array(points), (68,1,2))
        points1_transformed = cv2.transform(points1, tform)
        points_copy = np.reshape(points1_transformed, (68,2))
        transformed_coordinates = points_copy.tolist()

        return transformed_coordinates
    
    except IndexError:
        print("{} does not contain landmarks".format(image_name))
        pass

transformed_landmarks_set = []

for image in images:
    transformed_landmarks_set.append(image_processing(image))
```

    Processing headshots_kpo18c93cn001.jpg...
    Processing headshots_m59149oosppz.jpg...
    Processing headshots_r29qho5aia331.jpg...
    Processing headshots_xpo3mnqz0ex31.jpg...
    headshots_xpo3mnqz0ex31.jpg does not contain landmarks


Since the 4th image contains only a partial face, facial landmarks weren't constructed. Therefore, we need to exclude this image from our aggregation.


```python
del transformed_landmarks_set[-1]
```

Let's visualize the delauney triangulation for all selected images:


```python
transformed_images = []

for filename in sorted(os.listdir(os.getcwd())):
    if filename.startswith("transformed"):
        transformed_images.append(filename)

plt.figure(figsize=(40,100))
columns = 3
for i, image in enumerate(transformed_images):
    img = cv2.imread(image)
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    
    transformed_coordinates = transformed_landmarks_set[i]

    # calculate the 3rd order polygon for 68 points
    convexhull = cv2.convexHull(np.reshape(transformed_coordinates, (68,2)))

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(transformed_coordinates)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype = np.int32)
    
    for t in triangles:
        point1 = (t[0], t[1])
        point2 = (t[2], t[3])
        point3 = (t[4], t[5])

        # draw lines that connect point 1, 2, and 3 of all triangles
        cv2.line(img, point1, point2, (0,0,255), 5)
        cv2.line(img, point1, point3, (0,0,255), 5)
        cv2.line(img, point2, point3, (0,0,255), 5)
    
    plt.imshow(convertToRGB(img))

```


![png](Final%20Report/output_36_0.png)


In order to calculate the average face, we first need to generate the mean facial landmarks. This can be done by averaging the coordinates of all transformed landmarks:


```python
array = np.array(transformed_landmarks_set)
mean_face_points = np.mean(array, axis=0 , dtype = np.int32)
mean_face_points = mean_face_points.tolist()
```


```python
blank_image = np.zeros([600,600,3],dtype=np.uint8)
img.fill(255)

transformed_coordinates = mean_face_points

# calculate the 3rd order polygon for 68 points
convexhull = cv2.convexHull(np.reshape(transformed_coordinates, (68,2)))

# Delaunay triangulation
rect = cv2.boundingRect(convexhull)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(transformed_coordinates)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype = np.int32)

for t in triangles:
    point1 = (t[0], t[1])
    point2 = (t[2], t[3])
    point3 = (t[4], t[5])

    # draw lines that connect point 1, 2, and 3 of all triangles
    cv2.line(blank_image, point1, point2, (255,0,0), 5)
    cv2.line(blank_image, point1, point3, (255,0,0), 5)
    cv2.line(blank_image, point2, point3, (255,0,0), 5)
    
plt.imshow(blank_image)
```




    <matplotlib.image.AxesImage at 0x27b3c589780>




![png](Final%20Report/output_39_1.png)



```python
# codes adapted from https://github.com/spmallick/learnopencv/tree/master/FaceAverage

# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)
   
    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

   
    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array

    delaunayTri = []
    
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        

    
    return delaunayTri


def constrainPoint(p, w, h) :
    p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
    return p

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

imagesNorm = []
pointsNorm = transformed_landmarks_set
numImages = 3

for filename in sorted(os.listdir(os.getcwd())):
    if filename.startswith("transformed"):
        img = cv2.imread(filename)
        # Convert to floating point
        img = np.float32(img)/255.0
        imagesNorm.append(img)       
        
# Delaunay triangulation
rect = (0, 0, w, h)
dt = calculateDelaunayTriangles(rect, np.array(mean_face_points))

# Output image
output = np.zeros((h,w,3), np.float32())

# Warp input images to average image landmarks
for i in range(0, len(imagesNorm)) :
    img = np.zeros((h,w,3), np.float32())
    # Transform triangles one by one
    for j in range(0, len(dt)) :
        tin = []
        tout = []
            
        for k in range(0, 3) :                
            pIn = pointsNorm[i][dt[j][k]]
            pIn = constrainPoint(pIn, w, h)
            
            pOut = mean_face_points[dt[j][k]]
            pOut = constrainPoint(pOut, w, h)
                
            tin.append(pIn)
            tout.append(pOut)
            
        warpTriangle(imagesNorm[i], img, tin, tout)

    # save the warped images to file:
    cv2.imwrite(os.getcwd() + "\\warped_" + str(i) + ".jpg", img * 255)
    
    # Add image intensities for averaging
    output = output + img


```

# Final Output

After obtained the mean face points, we can warp the pixel inside the triangles of the input image to the average mesh triangles. Let's see how the input faces got warped:

## Before:


```python
plt.figure(figsize=(40,100))
columns = 3
for i, image in enumerate(transformed_images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(convertToRGB(cv2.imread(image)))
```


![png](Final%20Report/output_42_0.png)


## After:


```python
warped_images = []

for filename in sorted(os.listdir(os.getcwd())):
    if filename.startswith("warped"):
        warped_images.append(filename)

plt.figure(figsize=(40,100))
columns = 3
for i, image in enumerate(warped_images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(convertToRGB(cv2.imread(image)))
```


![png](Final%20Report/output_44_0.png)


We can see the faces have been morphed with respect to the position of the average landmarks. The morph is most pronounced in the last face due to the significant deviation from the average landmarks. After averaging the pixel from these images, we obtain the final average face:


```python
# Divide by numImages to get average
output = output / numImages

plt.figure(figsize=(40,40))
plt.imshow(convertToRGB(output))
```




    <matplotlib.image.AxesImage at 0x27b3ce08b00>




![png](Final%20Report/output_46_1.png)


# References:

Reimer, Vincent. (2018). The Average Faces of 42 Different Subreddits. Retrieved from URL

https://www.reddit.com/r/dataisbeautiful/comments/843zzy/the_average_faces_of_42_different_subreddits_oc/

Kazemi, V., & Sullivan, J. (2014). One millisecond face alignment with an ensemble of regression trees. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1867-1874).

Langlois, J. H., & Roggman, L. A. (1990). Attractive faces are only average. Psychological science, 1(2), 115-121.

Sagonas, C., Antonakos, E., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2016). 300 faces in-the-wild challenge: Database and results. Image and vision computing, 47, 3-18.

Valentine, T., Darling, S., & Donnelly, M. (2004). Why are average faces attractive? The effect of view and averageness on the attractiveness of female faces. Psychonomic Bulletin & Review, 11(3), 482-487.


