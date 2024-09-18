import cv2
import pygame
import numpy as np
import random
import math

def edge_detection_two(cap, screen, screen_width, screen_height):
    ret, frame = cap.read()
    if not ret:
        return

    # Convert the frame to grayscale and blur it to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Convert to RGB to allow colorization
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Colorize edges by applying random colors
    for i in range(edges_rgb.shape[0]):
        for j in range(edges_rgb.shape[1]):
            if np.array_equal(edges_rgb[i, j], [255, 255, 255]):  # Edge pixels are white
                # Assign bright colors to the edges
                edges_rgb[i, j] = [
                    random.randint(150, 255),  # Red
                    random.randint(150, 255),  # Green
                    random.randint(150, 255)   # Blue
                ]

    # Resize the image to fit the screen
    edges_resized = cv2.resize(edges_rgb, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    # Apply more exaggerated distortion effect (wavy edges)
    distortion_factor = abs(math.sin(pygame.time.get_ticks() / 500.0)) * 20  # Increase distortion
    for i in range(screen_width):
        if i % 10 == 0:  # Apply distortion every 10 pixels to reduce load
            edges_resized[:, i] = np.roll(edges_resized[:, i], int(distortion_factor), axis=0)

    # Apply a stronger glow effect (larger Gaussian blur)
    glow_effect = cv2.GaussianBlur(edges_resized, (15, 15), 0)

    # Blend the original distorted edges with the glow effect
    combined_effect = cv2.addWeighted(edges_resized, 0.5, glow_effect, 0.5, 0)

    # Swap axes to match Pygame's surface format
    edges_swapped = np.swapaxes(combined_effect, 0, 1)

    # Convert the NumPy array to a Pygame surface and display it
    surface = pygame.surfarray.make_surface(edges_swapped)
    screen.blit(surface, (0, 0))
    pygame.display.flip()