import os
import random
import cv2
import numpy as np
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import pygame
import time
import math

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
folder_path = 'obama'  # Update with your folder path
random_images = load_images_from_folder(folder_path)

# State for persistent face images (keeps same image on face for 2 seconds)
face_image_state = {
    'current_images': {},  # Dict mapping face index to assigned image
    'last_switch_time': time.time(),
    'image_hold_duration': 2.0,  # Hold same image for 2 seconds
    'last_beat_time': time.time(),
}

# Function to overlay random image with transparency onto the face bounding box
def overlay_random_image_with_alpha(background, overlay, x, y, w, h, scale_factor=1.5, offset_x=0, offset_y=0):
    # Increase the size of the bounding box by the scale factor
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Adjust the position - shift UP more to cover hair (face detection misses hair)
    # Center horizontally, but shift up by 30% of the height difference to cover hair
    new_x = x - (new_w - w) // 2 + offset_x
    vertical_hair_offset = int((new_h - h) * 0.35)  # Shift up more to cover hair
    new_y = y - (new_h - h) // 2 - vertical_hair_offset + offset_y

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
    if clipped_w <= 0 or clipped_h <= 0:
        return  # Skip if completely out of bounds
    
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
def face_replacement_effect(cap, screen, screen_width, screen_height, bpm=120):
    global face_image_state
    
    # Read the current frame from the webcam
    success, img = cap.read()
    
    if not success:
        return

    current_time = time.time()
    
    # Calculate beat timing for head bumping
    if bpm <= 0:
        bpm = 120  # Default BPM
    beat_duration = 60.0 / bpm  # Duration of one beat in seconds
    
    # Calculate diagonal offset for head bump effect
    # On each beat, the head jumps diagonally (down-right on beat, back to normal off-beat)
    beat_phase = (current_time % beat_duration) / beat_duration
    
    # Sharp bump: head goes down-right on the beat (first half), returns on second half
    bump_amount = 20  # pixels to move diagonally
    if beat_phase < 0.5:
        # On the beat - head bumps down and to the right
        offset_x = int(bump_amount * (1 - beat_phase * 2))  # Quick snap then ease back
        offset_y = int(bump_amount * (1 - beat_phase * 2))
    else:
        # Off-beat - head returns to center
        offset_x = 0
        offset_y = 0
    
    # Check if it's time to switch images (every 2 seconds)
    if current_time - face_image_state['last_switch_time'] >= face_image_state['image_hold_duration']:
        face_image_state['current_images'] = {}  # Clear to get new random images
        face_image_state['last_switch_time'] = current_time

    # Detect faces in the image
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected and replace the bounding boxes with random images
    if bboxs:
        for i, bbox in enumerate(bboxs):
            x, y, w, h = bbox['bbox']

            # Assign a persistent image to each face (by index)
            if i not in face_image_state['current_images']:
                face_image_state['current_images'][i] = random.choice(random_images)
            
            assigned_image = face_image_state['current_images'][i]
            
            # Replace the face bounding box with the assigned image, with diagonal bump offset
            overlay_random_image_with_alpha(img, assigned_image, x, y, w, h, scale_factor=2, offset_x=offset_x, offset_y=offset_y)

    # Resize the frame to fit the Pygame screen dimensions
    img_resized = cv2.resize(img, (screen_width, screen_height))

    # Convert the OpenCV image (BGR) to the Pygame surface (RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = np.swapaxes(img_rgb, 0, 1)

    face_surface = pygame.surfarray.make_surface(img_rgb)

    # Display the face detection result in Pygame
    screen.blit(pygame.transform.scale(face_surface, (screen_width, screen_height)), (0, 0))

    return img
