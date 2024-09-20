import os
import random
import cv2
import numpy as np
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import pygame

# Initialize the FaceDetector object
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Load all images from the specified folder with alpha channel (transparency)
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)  # Load with alpha channel
        if img is not None:
            images.append(img)
    return images

# Assuming your images are stored in a folder called 'images'
folder_path = 'musk'  # Update with your folder path
random_images = load_images_from_folder(folder_path)

# Function to overlay random image with transparency onto the face bounding box
def overlay_random_image_with_alpha(background, overlay, x, y, w, h, scale_factor=1.5):
    # Increase the size of the bounding box by the scale factor
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Adjust the position so the image stays centered
    new_x = x - (new_w - w) // 2
    new_y = y - (new_h - h) // 2

    # Resize the random image to fill the enlarged bounding box
    overlay_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Extract the BGR and alpha channels from the overlay image
    if overlay_resized.shape[2] == 4:  # If the image has an alpha channel
        overlay_bgr = overlay_resized[:, :, :3]  # First three channels are BGR
        alpha_mask = overlay_resized[:, :, 3] / 255.0  # Normalize alpha channel to range [0, 1]
    else:
        overlay_bgr = overlay_resized
        alpha_mask = np.ones((new_h, new_w), dtype=float)  # If no alpha, assume full opacity

    # Clip the bounding box coordinates to ensure they don't go outside the background image
    x1, y1 = max(0, new_x), max(0, new_y)
    x2, y2 = min(background.shape[1], new_x + new_w), min(background.shape[0], new_y + new_h)

    # Adjust the overlay if part of it goes outside the background dimensions
    clipped_w, clipped_h = x2 - x1, y2 - y1
    overlay_resized = cv2.resize(overlay_bgr, (clipped_w, clipped_h), interpolation=cv2.INTER_AREA)
    alpha_mask = cv2.resize(alpha_mask, (clipped_w, clipped_h), interpolation=cv2.INTER_AREA)

    # Get the region of interest (ROI) from the background where the overlay will be placed
    roi = background[y1:y2, x1:x2]

    # Blend the overlay with the ROI using the alpha mask
    for c in range(0, 3):  # For each BGR channel
        roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + overlay_resized[:, :, c] * alpha_mask

    # Place the blended result back into the original background image
    background[y1:y2, x1:x2] = roi

# Face detection effect function with random image replacement and alpha handling
def face_replacement_effect(cap, screen, screen_width, screen_height):
    # Read the current frame from the webcam
    success, img = cap.read()
    
    if not success:
        return

    # Detect faces in the image
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected and replace the bounding boxes with random images
    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox['bbox']

            # Replace the face bounding box with a larger random image from the folder
            random_image = random.choice(random_images)
            overlay_random_image_with_alpha(img, random_image, x, y, w, h, scale_factor=2)

    # Resize the frame to fit the Pygame screen dimensions
    img_resized = cv2.resize(img, (screen_width, screen_height))

    # Convert the OpenCV image (BGR) to the Pygame surface (RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = np.swapaxes(img_rgb, 0, 1)

    face_surface = pygame.surfarray.make_surface(img_rgb)

    # Display the face detection result in Pygame
    screen.blit(pygame.transform.scale(face_surface, (screen_width, screen_height)), (0, 0))

    return img
