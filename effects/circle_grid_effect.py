import cv2
import numpy as np
import pygame
import random

def circle_grid_effect(bpm, previous_frame, cap, screen, screen_width, screen_height):
    # Set the dimensions for the circle grid window (smaller than the full screen)
    grid_window_width = screen_width // 2  # 1/3 of the screen width
    grid_window_height = screen_height // 2  # 1/3 of the screen height

    # Cell size for the grid
    cell_width, cell_height = 10, 10
    new_width, new_height = int(grid_window_width / cell_width), int(grid_window_height / cell_height)

    # Define color intensity adjustment factor based on BPM
    color_intensity_factor = np.interp(bpm, [60, 180], [0.5, 1.5])  # Adjust between 0.5 and 1.5 based on BPM

    # Generate a random or BPM-dependent RGB tint
    tint_color = np.array([random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)], dtype=np.uint16)

    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        return previous_frame  # Return previous frame if no new frame is captured

    # Resize the frame to fit the grid window
    small_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Create a black window for drawing the grid
    black_window = np.zeros((grid_window_height, grid_window_width, 3), np.uint8)

    # Loop over the grid and draw circles based on the resized webcam frame
    for i in range(new_height):
        for j in range(new_width):
            color = small_image[i, j]

            # Apply color intensity adjustment
            B = int(color[0] * color_intensity_factor)
            G = int(color[1] * color_intensity_factor)
            R = int(color[2] * color_intensity_factor)

            # Apply the tint color by multiplying the current pixel's color by the tint
            B = (B * tint_color[0] / 255)
            G = (G * tint_color[1] / 255)
            R = (R * tint_color[2] / 255)

            # Ensure the values are within valid RGB range (0-255)
            B, G, R = [int(np.clip(B, 0, 255)), int(np.clip(G, 0, 255)), int(np.clip(R, 0, 255))]

            coord = (j * cell_width + cell_width // 2, i * cell_height + cell_height // 2)

            # Draw circles on the black window
            cv2.circle(black_window, coord, 5, (B, G, R), 2)

    # Add the afterimage effect by blending the current frame with the previous frame
    if previous_frame is not None:
        # Blend the current frame with the previous frame (create a fading effect)
        black_window = cv2.addWeighted(black_window, 0.8, previous_frame, 0.2, 0)  # 0.8 current, 0.2 previous

    # Update the previous frame
    previous_frame = black_window.copy()

    # Convert the black window (OpenCV format) to Pygame-compatible format
    black_window_rgb = cv2.cvtColor(black_window, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(black_window_rgb, (grid_window_width, grid_window_height), interpolation=cv2.INTER_LINEAR)

    # Calculate the offset to center the grid window in the middle of the screen
    x_offset = (screen_width - grid_window_width) // 2
    y_offset = (screen_height - grid_window_height) // 2

    # Create a Pygame surface from the resized image
    surface = pygame.surfarray.make_surface(np.swapaxes(resized_frame, 0, 1))

    # Blit the surface onto the Pygame screen at the calculated offset
    screen.blit(surface, (x_offset, y_offset))
    pygame.display.flip()

    return previous_frame  # Return the updated previous frame
