import dlib
import cv2
import numpy as np

# Paths to the input image and the shape predictor model
image_path = "/Users/dush/Documents/FaceDetection/face3.jpg"  # Path to the input image (string)
shape_predictor_path = "/Users/dush/Documents/FaceDetection/shape_predictor_68_face_landmarks.dat"  # Path to the shape predictor model (string)

# Load the pre-trained face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()  # Initialize the dlib face detector (function)
predictor = dlib.shape_predictor(shape_predictor_path)  # Initialize the dlib shape predictor for facial landmarks (object)

# Load the input image using OpenCV
image = cv2.imread(image_path)  # Load the input image using OpenCV (numpy array)

# Convert the input image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale for processing (numpy array)

# Detect faces in the grayscale image using the face detector
faces = detector(gray)  # Detect faces in the grayscale image using the dlib face detector (list of dlib rectangles)

# Iterate over each detected face
for face in faces:
    # Predict the facial landmarks for the current face using the landmark predictor
    landmarks = predictor(gray, face)  # Predict facial landmarks for the current face using the dlib shape predictor (dlib shape object)

    # Extract the landmarks corresponding to the left eye
    left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])  # Extract landmark points for the left eye (numpy array)
    # Extract the landmarks corresponding to the right eye
    right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])  # Extract landmark points for the right eye (numpy array)
    # Extract the landmarks corresponding to the left eyebrow
    left_eyebrow_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)])  # Extract landmark points for the left eyebrow (numpy array)
    # Extract the landmarks corresponding to the right eyebrow
    right_eyebrow_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)])  # Extract landmark points for the right eyebrow (numpy array)

    # Combine eye and eyebrow points for each eye
    left_eye_and_eyebrow_pts = np.concatenate((left_eye_pts, left_eyebrow_pts))  # Combine points for the left eye and eyebrow (numpy array)
    right_eye_and_eyebrow_pts = np.concatenate((right_eye_pts, right_eyebrow_pts))  # Combine points for the right eye and eyebrow (numpy array)

    # Calculate the bounding rectangle around the left eye and eyebrow
    left_eye_and_eyebrow_rect = cv2.boundingRect(left_eye_and_eyebrow_pts)  # Calculate bounding rectangle around left eye and eyebrow (tuple of integers)
    # Calculate the bounding rectangle around the right eye and eyebrow
    right_eye_and_eyebrow_rect = cv2.boundingRect(right_eye_and_eyebrow_pts)  # Calculate bounding rectangle around right eye and eyebrow (tuple of integers)

    # Extract the region of interest (ROI) around the left eye and eyebrow from the original image
    left_eye_and_eyebrow_roi = image[left_eye_and_eyebrow_rect[1]:left_eye_and_eyebrow_rect[1] + left_eye_and_eyebrow_rect[3],
                         left_eye_and_eyebrow_rect[0]:left_eye_and_eyebrow_rect[0] + left_eye_and_eyebrow_rect[2]]  # Extract ROI around left eye and eyebrow (numpy array)
    # Extract the region of interest (ROI) around the right eye and eyebrow from the original image
    right_eye_and_eyebrow_roi = image[right_eye_and_eyebrow_rect[1]:right_eye_and_eyebrow_rect[1] + right_eye_and_eyebrow_rect[3],
                          right_eye_and_eyebrow_rect[0]:right_eye_and_eyebrow_rect[0] + right_eye_and_eyebrow_rect[2]]  # Extract ROI around right eye and eyebrow (numpy array)

    # Draw rectangle around the left eye and eyebrow on the original image
    cv2.rectangle(image, (left_eye_and_eyebrow_rect[0], left_eye_and_eyebrow_rect[1]),
                  (left_eye_and_eyebrow_rect[0] + left_eye_and_eyebrow_rect[2], left_eye_and_eyebrow_rect[1] + left_eye_and_eyebrow_rect[3]), (255, 0, 0), 2)  # Draw rectangle around left eye and eyebrow
    # Draw rectangle around the right eye and eyebrow on the original image
    cv2.rectangle(image, (right_eye_and_eyebrow_rect[0], right_eye_and_eyebrow_rect[1]),
                  (right_eye_and_eyebrow_rect[0] + right_eye_and_eyebrow_rect[2], right_eye_and_eyebrow_rect[1] + right_eye_and_eyebrow_rect[3]), (255, 0, 0), 2)  # Draw rectangle around right eye and eyebrow

    # Display the original image with drawn eye and eyebrow rectangles
    cv2.imshow("Image", image)  # Display the image with drawn rectangles (None)
    cv2.waitKey(0)  # Wait for a key press (int)


