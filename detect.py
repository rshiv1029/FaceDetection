from sys import argv
import cv2
from cv2 import CascadeClassifier, imread, imshow, cvtColor, rectangle, waitKey, COLOR_BGR2GRAY, CASCADE_SCALE_IMAGE

# Read in user inputs
numInputs = argv[0]
imagePath = argv[1]

# Select & create cascade: A Cascade is just an xml that contains data on how to detect the face 
cascade_xml = 'haarcascade_frontalface_default.xml'
cascade = CascadeClassifier(cascade_xml)

# Read the image and convert to grayscale  
image = imread(imagePath)
grayscale_image = cvtColor(image, COLOR_BGR2GRAY) 

# Detect faces in the image
faces = cascade.detectMultiScale(
    grayscale_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = CASCADE_SCALE_IMAGE
)

print("Number of faces found: ", len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

imshow("Faces found", image)
waitKey(0)