import dlib
import cv2
import numpy as np
import os

# Paths to the shape predictor model
shape_predictor_path = "/Users/agastyabhardwaj/Downloads/shape_predictor_68_face_landmarks.dat"  # Path to the shape predictor model (string)

# Load the pre-trained face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()  # Initialize the dlib face detector (function)
predictor = dlib.shape_predictor(shape_predictor_path)  # Initialize the dlib shape predictor for facial landmarks (object)

face_file_path = "/Users/agastyabhardwaj/Downloads/utkcropped/utkcropped"
dir_list = os.listdir(face_file_path)

for num in dir_list:
    try:
        image_path = "/Users/agastyabhardwaj/Downloads/utkcropped/utkcropped/" + num
        image = cv2.imread(image_path)  # Load the input image using OpenCV (numpy array)

        # Convert the input image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale for processing (numpy array)

        # Detect faces in the grayscale image using the face detector
        faces = detector(gray)


        for face in faces:
            # Predict the facial landmarks for the current face using the landmark predictor
            landmarks = predictor(gray, face)  # Predict facial landmarks for the current face using the dlib shape predictor (dlib shape object)

            # Extract the landmarks corresponding to the left eye
            mouth_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])  # Extract landmark points for the left eye (numpy array)

            # Calculate the bounding rectangle around the left eye and eyebrow
            mouth_rect = cv2.boundingRect(mouth_pts)  # Calculate bounding rectangle around left eye and eyebrow (tuple of integers)

            # Extract the region of interest (ROI) around the left eye and eyebrow from the original image
            mouth_roi = image[mouth_rect[1]:mouth_rect[1] + mouth_rect[3],
                                 mouth_rect[0]:mouth_rect[0] + mouth_rect[2]]  # Extract ROI around left eye and eyebrow (numpy array)

            # Draw rectangle around the left eye and eyebrow on the original image
            cv2.rectangle(image, (mouth_rect[0], mouth_rect[1]),
                          (mouth_rect[0] + mouth_rect[2], mouth_rect[1] + mouth_rect[3]), (255, 0, 0), 2)  # Draw rectangle around left eye and eyebrow
            mouth_y_range = (mouth_rect[0], mouth_rect[0] + mouth_rect[2])
            mouth_x_range = (mouth_rect[1], mouth_rect[1] + mouth_rect[3])
            cutImage = image[mouth_x_range[0]:mouth_x_range[1], mouth_y_range[0]:mouth_y_range[1]]
            x = num.split("_")
            if x[2] == "0":
                cv2.imwrite(f"/Users/agastyabhardwaj/Downloads/ethnicitymouth/White/{num}", cutImage)
            if x[2] == "1":
                cv2.imwrite(f"/Users/agastyabhardwaj/Downloads/ethnicitymouth/Black/{num}", cutImage)
            if x[2] == "2":
                cv2.imwrite(f"/Users/agastyabhardwaj/Downloads/ethnicitymouth/Asian/{num}", cutImage)
            if x[2] == "3":
                cv2.imwrite(f"/Users/agastyabhardwaj/Downloads/ethnicitymouth/Indian/{num}", cutImage)
            if x[2] == "4":
                cv2.imwrite(f"/Users/agastyabhardwaj/Downloads/ethnicitymouth/Others/{num}", cutImage)
    except:
        pass
