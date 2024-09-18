import cv2
import numpy as np
import pygame
import random
import time

def ascii_webcam_effect(bpm, last_beat_time, cap, screen, screen_width, screen_height, black, color_change_interval=1):
    chars = np.asarray(list(' .,:irs?@9B&#BUSH'))  # Define the ASCII characters

    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        return last_beat_time

    # Resize the frame to a smaller size for the ASCII grid
    ascii_width = 100  # Limit the width to 100 characters
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    ascii_height = int(aspect_ratio * ascii_width * 0.55)  # Adjust height based on the aspect ratio of ASCII characters

    # Resize the webcam frame to the target ASCII dimensions
    resized_frame = cv2.resize(frame, (ascii_width, ascii_height))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Normalize the grayscale image to a range between 0 and the number of ASCII characters
    normalized_gray = (gray / 255) * (chars.size - 1)

    # Convert the grayscale values to ASCII characters
    ascii_image = chars[normalized_gray.astype(int)]

    # Prepare font and render the ASCII string character by character
    font = pygame.font.SysFont('Courier', 12)  # Adjust the font size as needed
    screen.fill(black)

    # Calculate the center position for the ASCII art
    x_offset = (screen_width - ascii_width * 10) // 2  # Center horizontally
    y_offset = (screen_height - ascii_height * 12) // 2  # Center vertically

    current_time = time.time()
    beat_interval = 60.0 / bpm

    # Initialize colored_indices so it always has a value
    total_chars = ascii_width * ascii_height
    num_colored_chars = int(0.2 * total_chars)  # 20% of the total characters
    colored_indices = []

    # Change colors based on a slower, fixed interval rather than the beat interval
    if current_time - last_beat_time >= color_change_interval:
        colored_indices = random.sample(range(total_chars), num_colored_chars)
        last_beat_time = current_time

    # Render the ASCII art onto the screen
    for i, line in enumerate(ascii_image):
        for j, char in enumerate(line):
            idx = i * len(line) + j

            # Check if this character should be colored
            if idx in colored_indices:
                # Generate a random color
                random_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                # Render the character with a colored background
                text = font.render(char, True, (0, 0, 0), random_color)  # Black text with colored background
            else:
                # Render the character in white with no background
                text = font.render(char, True, (255, 255, 255))

            # Render each character with offset (centered window)
            screen.blit(text, (x_offset + j * 10, y_offset + i * 12))  # Adjust based on character width and line height

    pygame.display.flip()  # Update the screen

    return last_beat_time
