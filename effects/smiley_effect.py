import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import numpy
import pygame

# Initialize the FaceDetector object
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Function to draw a smiley face inside a bounding box
def draw_smiley(img, x, y, w, h):
    # Center of the face bounding box
    center_x, center_y = x + w // 2, y + h // 2

    # Draw the face (head)
    cv2.circle(img, (center_x, center_y), min(w, h) // 2, (0, 255, 255), -1)  # Yellow face

    # Draw the eyes
    eye_radius = min(w, h) // 10
    eye_offset_x = w // 4
    eye_offset_y = h // 6
    cv2.circle(img, (center_x - eye_offset_x, center_y - eye_offset_y), eye_radius, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (center_x + eye_offset_x, center_y - eye_offset_y), eye_radius, (0, 0, 0), -1)  # Right eye

    # Draw the smile (arc/ellipse)
    smile_thickness = 2
    smile_radius_x = w // 5
    smile_radius_y = h // 10
    smile_start_angle = 20
    smile_end_angle = 160
    cv2.ellipse(img, (center_x, center_y + h // 6), (smile_radius_x, smile_radius_y),
                0, smile_start_angle, smile_end_angle, (0, 0, 0), smile_thickness)

# Face detection effect function with smileys
def face_detection_effect(cap, screen, screen_width, screen_height):
    # Read the current frame from the webcam
    success, img = cap.read()
    
    if not success:
        return

    # Detect faces in the image
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected and draw bounding boxes and smileys
    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            
            # Draw the smiley face inside the bounding box
            draw_smiley(img, x, y, w, h)

            # Optionally: Draw the confidence score and bounding box around the face
            score = int(bbox['score'][0] * 100)
            cvzone.putTextRect(img, f'{score}%', (x, y - 10))
            cvzone.cornerRect(img, (x, y, w, h), colorR=(0, 255, 0))

    # Resize the frame to fit the Pygame screen dimensions
    img_resized = cv2.resize(img, (screen_width, screen_height))

    # Convert the OpenCV image (BGR) to the Pygame surface (RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = numpy.swapaxes(img_rgb, 0, 1)

    face_surface = pygame.surfarray.make_surface(img_rgb)

    # Display the face detection result in Pygame
    screen.blit(pygame.transform.scale(face_surface, (screen_width, screen_height)), (0, 0))

    return img
