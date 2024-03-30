import dlib
import cv2
import numpy as np

# Paths to the input image and the shape predictor model
image_path = "/Users/dush/Documents/FaceDetection/face.jpg"  # File path to the input image
shape_predictor_path = "/Users/dush/Documents/FaceDetection/shape_predictor_68_face_landmarks.dat"  # File path to the shape predictor model

# Load the pre-trained face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()  # Initialize the face detector
predictor = dlib.shape_predictor(shape_predictor_path)  # Initialize the facial landmark predictor

# Load the input image using OpenCV
image = cv2.imread(image_path)  # Load the input image

# Convert the input image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

# Detect faces in the grayscale image using the face detector
faces = detector(gray)  # Detect faces in the grayscale image

# Iterate over each detected face
for face in faces:
    # Predict the facial landmarks for the current face using the landmark predictor
    landmarks = predictor(gray, face)  # Predict facial landmarks for the current face

    # Extract the landmarks corresponding to the left eye
    left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in
                             range(36, 42)])  # List of (x, y) coordinates for the left eye

    # Extract the landmarks corresponding to the right eye
    right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in
                              range(42, 48)])  # List of (x, y) coordinates for the right eye

    # Extract the landmarks corresponding to the left eyebrow
    left_eyebrow_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in
                                 range(17, 22)])  # List of (x, y) coordinates for the left eyebrow

    # Extract the landmarks corresponding to the right eyebrow
    right_eyebrow_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in
                                  range(22, 27)])  # List of (x, y) coordinates for the right eyebrow

    # Combine eye and eyebrow points for each eye
    left_eye_and_eyebrow_pts = np.concatenate(
        (left_eye_pts, left_eyebrow_pts))  # Combined points for the left eye and eyebrow
    right_eye_and_eyebrow_pts = np.concatenate(
        (right_eye_pts, right_eyebrow_pts))  # Combined points for the right eye and eyebrow

    # Calculate the bounding rectangle around the left eye and eyebrow
    left_eye_and_eyebrow_rect = cv2.boundingRect(
        left_eye_and_eyebrow_pts)  # Bounding rectangle around the left eye and eyebrow

    # Calculate the bounding rectangle around the right eye and eyebrow
    right_eye_and_eyebrow_rect = cv2.boundingRect(
        right_eye_and_eyebrow_pts)  # Bounding rectangle around the right eye and eyebrow

    # Calculate the combined bounding rectangle around both eyes and eyebrows
    combined_rect = (
        min(left_eye_and_eyebrow_rect[0], right_eye_and_eyebrow_rect[0]),  # X-coordinate of the top-left corner
        min(left_eye_and_eyebrow_rect[1], right_eye_and_eyebrow_rect[1]),  # Y-coordinate of the top-left corner
        max(left_eye_and_eyebrow_rect[0] + left_eye_and_eyebrow_rect[2],  # Width of the rectangle
            right_eye_and_eyebrow_rect[0] + right_eye_and_eyebrow_rect[2]) -
        min(left_eye_and_eyebrow_rect[0], right_eye_and_eyebrow_rect[0]),  # Height of the rectangle
        max(left_eye_and_eyebrow_rect[1] + left_eye_and_eyebrow_rect[3],  # Height of the rectangle
            right_eye_and_eyebrow_rect[1] + right_eye_and_eyebrow_rect[3]) -
        min(left_eye_and_eyebrow_rect[1], right_eye_and_eyebrow_rect[1])  # Height of the rectangle
    )

    # Draw rectangle around both eyes and eyebrows on the original image
    cv2.rectangle(image, (combined_rect[0], combined_rect[1]),
                  (combined_rect[0] + combined_rect[2], combined_rect[1] + combined_rect[3]), (255, 0, 0), 2)

    # Display the original image with drawn eye and eyebrow rectangle
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# Close all OpenCV windows after processing
cv2.destroyAllWindows()
