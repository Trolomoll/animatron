import cv2
import numpy as np
import pygame
import random
import time

# Store previous frame for motion detection
prev_gray_frame = None
# Store current motion color
current_motion_color = (255, 100, 100)
last_color_change_time = 0

def ascii_webcam_effect(bpm, last_beat_time, cap, screen, screen_width, screen_height, black, color_change_interval=1):
    global prev_gray_frame, current_motion_color, last_color_change_time
    
    chars = np.asarray(list(' .,:irs?@9B&#BUSH'))  # Define the ASCII characters

    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        return last_beat_time

    # Calculate character size to fill the screen
    # Use a font size that scales with screen size
    font_size = max(8, min(screen_width // 120, screen_height // 80))
    char_width = font_size * 0.6  # Approximate width of monospace character
    char_height = font_size * 1.2  # Line height
    
    # Calculate how many characters fit on screen
    ascii_width = int(screen_width / char_width)
    ascii_height = int(screen_height / char_height)

    # Resize the webcam frame to the target ASCII dimensions
    resized_frame = cv2.resize(frame, (ascii_width, ascii_height))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Motion detection - compare with previous frame
    if prev_gray_frame is None or prev_gray_frame.shape != gray.shape:
        prev_gray_frame = gray.copy()
    
    # Calculate difference between frames to detect motion
    frame_diff = cv2.absdiff(prev_gray_frame, gray)
    motion_threshold = 20  # Threshold for motion detection
    motion_mask = frame_diff > motion_threshold
    
    # Update previous frame
    prev_gray_frame = gray.copy()

    # Normalize the grayscale image to a range between 0 and the number of ASCII characters
    normalized_gray = (gray / 255) * (chars.size - 1)

    # Convert the grayscale values to ASCII characters
    ascii_image = chars[normalized_gray.astype(int)]

    # Prepare font and render the ASCII string character by character
    font = pygame.font.SysFont('Courier', font_size)
    screen.fill(black)

    # Start from top-left to fill the whole screen
    x_offset = 0
    y_offset = 0

    current_time = time.time()
    beat_interval = 60.0 / bpm
    
    # Strong, vibrant colors
    strong_colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
    ]
    
    # Change the motion color every few beats (e.g., every 4 beats)
    color_change_interval_seconds = beat_interval * 4
    if current_time - last_color_change_time >= color_change_interval_seconds:
        current_motion_color = random.choice(strong_colors)
        last_color_change_time = current_time
    
    if current_time - last_beat_time >= beat_interval:
        last_beat_time = current_time

    # Render the ASCII art onto the screen
    for i, line in enumerate(ascii_image):
        for j, char in enumerate(line):
            # Check if this character has motion detected
            if motion_mask[i, j]:
                # Use the same color for all moving characters
                text = font.render(char, True, (0, 0, 0), current_motion_color)
            else:
                # Render the character in white with no background (static areas)
                text = font.render(char, True, (255, 255, 255))

            # Render each character to fill the screen
            screen.blit(text, (x_offset + j * char_width, y_offset + i * char_height))

    pygame.display.flip()  # Update the screen

    return last_beat_time
