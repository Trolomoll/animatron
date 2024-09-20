import cv2
import pygame
import numpy as np

def edge_detection(cap, screen, screen_width, screen_height):
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_resized = cv2.resize(edges_rgb, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
    edges_swapped = np.swapaxes(edges_resized, 0, 1)

    surface = pygame.surfarray.make_surface(edges_swapped)
    screen.blit(surface, (0, 0))
    pygame.display.flip()


import cv2
import numpy as np
import pygame
import time

# Initialize a global variable to store the last beat time and first frame
previous_edges = None
last_beat_time = 0

def edge_detection_with_pulse_and_afterimage(cap, screen, screen_width, screen_height, bpm, afterimage_alpha=0.9):
    global previous_edges, last_beat_time

    # Ensure BPM is at least 1 to avoid division by zero
    bpm = max(bpm, 1)
    beat_interval = 60.0 / bpm

    # Capture a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        return

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate the edges to create a spreading effect (pulse effect)
    dilation_size = 2
    dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=1)

    # Convert the dilated edge-detected frame to RGB
    dilated_edges_rgb = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2RGB)

    # Get the current time
    current_time = time.time()

    # If a beat occurred, increase the dilation (pulse the edges outward)
    if current_time - last_beat_time >= beat_interval:
        dilation_size = 5
        dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)
        dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=1)
        dilated_edges_rgb = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2RGB)
        last_beat_time = current_time

    # Resize the dilated edge-detected frame to match the screen dimensions
    dilated_edges_resized = cv2.resize(dilated_edges_rgb, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    # Swap the axes to match the format expected by Pygame
    dilated_edges_swapped = np.swapaxes(dilated_edges_resized, 0, 1)

    # If a previous frame exists, blend it with the current frame (afterimage effect)
    if previous_edges is not None:
        dilated_edges_swapped = cv2.addWeighted(previous_edges, afterimage_alpha, dilated_edges_swapped, 1 - afterimage_alpha, 0)

    # Update the previous frame with the current frame
    previous_edges = dilated_edges_swapped.copy()

    # Convert the frame to a Pygame surface and display it
    surface = pygame.surfarray.make_surface(dilated_edges_swapped)
    screen.blit(surface, (0, 0))
    pygame.display.flip()




def edge_detection_with_afterimage(cap, screen, screen_width, screen_height, afterimage_alpha=0.8):
    global previous_edges
    
    # Capture a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        return
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Convert the edge-detected frame to RGB
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Resize the edge-detected frame to match the screen dimensions
    edges_resized = cv2.resize(edges_rgb, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
    
    # Swap the axes to match the format expected by Pygame
    edges_swapped = np.swapaxes(edges_resized, 0, 1)
    
    # If a previous frame exists, blend it with the current frame
    if previous_edges is not None:
        edges_swapped = cv2.addWeighted(previous_edges, afterimage_alpha, edges_swapped, 1 - afterimage_alpha, 0)
    
    # Update the previous frame with the current frame
    previous_edges = edges_swapped.copy()
    
    # Convert the frame to a Pygame surface and display it
    surface = pygame.surfarray.make_surface(edges_swapped)
    screen.blit(surface, (0, 0))
    pygame.display.flip()
