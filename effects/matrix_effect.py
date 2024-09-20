import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from random import randint
import time

import pygame

# Initialize necessary components
segmenter = SelfiSegmentation(1)

# Constants
cell_width, cell_height = 10, 15
chars = " .,-~:;=!*#$@"
norm = 255 / len(chars)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.4

# Initialize font speed and frame rate tracking
font_speed = 0
fps_start_time = time.time()

# Function to draw matrix-like background
def matrix_background(matrix_window, i, j, font_speed):
    matrix_text = randint(0, 1)
    cv2.putText(matrix_window, str(matrix_text), (j * cell_width, i * cell_height + font_speed),
                font, font_size, (80, 120 - i * 3, 0), 1)

# Function to automatically correct the orientation of the frame
def correct_orientation(frame):
    # Check the dimensions of the frame
    (h, w) = frame.shape[:2]

    # If height is greater than width, it means the image might be rotated
    if h > w:
        # Rotate the frame 90 degrees counterclockwise to correct the orientation
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame

# Matrix effect function
def matrix_effect(cap, screen, screen_width, screen_height):
    global font_speed, fps_start_time

    # Initialize the matrix window
    matrix_window = np.zeros((screen_height, screen_width, 3), np.uint8)

    # Capture frame from the camera
    ret, frame = cap.read()
    if not ret:
        return

    # Correct the orientation of the frame
    frame = correct_orientation(frame)

    # Resize the image and apply background removal
    small_image = cv2.resize(
        frame,
        (int(screen_width / cell_width), int(screen_height / cell_height)),
        interpolation=cv2.INTER_NEAREST)
    small_image = segmenter.removeBG(small_image, (0, 0, 0))
    gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

    # Loop through each pixel and map intensity to a character
    for i in range(int(screen_height / cell_height)):
        for j in range(int(screen_width / cell_width)):
            intensity = gray_image[i, j]
            char_index = int(intensity / norm)
            color = small_image[i, j]
            G = int(color[1])

            # If background (black), draw matrix background text
            if color[1] == 0:
                if (i * cell_height + font_speed) < screen_height:
                    matrix_background(matrix_window, i, j, font_speed)
                else:
                    font_speed = 0
            else:
                char = chars[char_index]
                cv2.putText(matrix_window, char,
                            (j * cell_width + 5, i * cell_height + 12), font,
                            font_size, (0, G, 0), 1)

    font_speed += 1

    # Calculate FPS and display it
    fps_end_time = time.time()
    fps = 1 / (fps_end_time - fps_start_time)
    fps_start_time = fps_end_time
    fps_text = "FPS: {:.1f}".format(fps)
    cv2.putText(matrix_window, fps_text, (10, 20), font, 0.5, (255, 0, 255), 1)

    # Display matrix effect on the screen using Pygame
    swapped_matrix_window = np.swapaxes(matrix_window, 0, 1)

    matrix_surface = pygame.surfarray.make_surface(swapped_matrix_window)
    screen.blit(
        pygame.transform.scale(matrix_surface, (screen_width, screen_height)),
        (0, 0))

    return swapped_matrix_window
