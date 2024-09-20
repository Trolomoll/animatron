import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import pygame
import time

# Initialize the FaceDetector object
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Function to compute smiley color based on BPM and beat timing
def get_smiley_color(bpm, last_beat_time):
    current_time = time.time()
    beat_interval = 60.0 / bpm
    time_since_last_beat = (current_time - last_beat_time) % beat_interval
    beat_progress = time_since_last_beat / beat_interval  # Between 0 and 1

    # Interpolate color between yellow (0) and red (1)
    start_color = np.array([0, 255, 255])  # Yellow
    end_color = np.array([0, 0, 255])  # Red
    color = (1 - beat_progress) * start_color + beat_progress * end_color

    # Ensure color is a tuple of integers (required by OpenCV)
    return tuple(map(int, color))

# Function to draw a smiley face inside a bounding box with dynamic color
def draw_smiley(img, x, y, w, h, smiley_color):
    # Center of the face bounding box
    center_x, center_y = x + w // 2, y + h // 2

    # Draw the face (head) with the dynamic smiley color
    cv2.circle(img, (center_x, center_y), min(w, h) // 2, smiley_color, -1)

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

# Face detection effect function with smileys that change color based on BPM
def face_detection_effect_with_bpm(cap, screen, screen_width, screen_height, bpm, last_beat_time):
    # Read the current frame from the webcam
    success, img = cap.read()
    
    if not success:
        return last_beat_time

    # Detect faces in the image
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected and draw bounding boxes and smileys
    if bboxs:
        smiley_color = get_smiley_color(bpm, last_beat_time)  # Get the dynamic color based on BPM
        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            
            # Draw the smiley face inside the bounding box with the dynamic color
            draw_smiley(img, x, y, w, h, smiley_color)

    # Resize the frame to fit the Pygame screen dimensions
    img_resized = cv2.resize(img, (screen_width, screen_height))

    # Convert the OpenCV image (BGR) to the Pygame surface (RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = np.swapaxes(img_rgb, 0, 1)

    face_surface = pygame.surfarray.make_surface(img_rgb)

    # Display the face detection result in Pygame
    screen.blit(pygame.transform.scale(face_surface, (screen_width, screen_height)), (0, 0))

    return last_beat_time
