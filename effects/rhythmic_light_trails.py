import cv2
import pygame
import random
import math
import numpy as np

def rhythmic_light_trails(cap, screen, screen_width, screen_height, black, bpm):
    ret, frame1 = cap.read()
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.GaussianBlur(frame1_gray, (11, 11), 0)  # Less blur for more detail
    ret, frame2 = cap.read()
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (11, 11), 0)

    delta_frame = cv2.absdiff(frame1_gray, frame2_gray)
    thresh = cv2.threshold(delta_frame, 15, 255, cv2.THRESH_BINARY)[1]  # Lower threshold for more sensitivity
    thresh = cv2.dilate(thresh, None, iterations=1)  # Less dilation for finer detail

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set colors based on BPM (warmer colors for faster BPMs)
    if bpm < 80:
        trail_color = (0, 128, 255)  # Cool Blue
    elif bpm < 120:
        trail_color = (0, 255, 128)  # Green
    else:
        trail_color = (255, 128, 0)  # Warm Orange

    # Smaller pulse size for finer balls
    base_size = 2 + (bpm / 40)  # Even smaller base size

    # Screen trail effect background (slight fade out)
    overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 20))  # Slightly slower fade for longer trails
    screen.blit(overlay, (0, 0))

    if len(contours) > 0:
        # If motion is detected, generate glowing trails that pulse with BPM
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Much lower threshold - catch smaller movements
                continue
            
            # Sample multiple points along the contour for more balls
            contour_points = contour.reshape(-1, 2)
            
            # Take every Nth point to create multiple balls along the contour edge
            step = max(1, len(contour_points) // 15)  # Up to ~15 balls per contour
            
            for i in range(0, len(contour_points), step):
                px, py = contour_points[i]
                
                # Scale coordinates to screen size
                screen_x = int(px * screen_width / frame1.shape[1])
                screen_y = int(py * screen_height / frame1.shape[0])

                # Create pulsing light trails with variation
                pulse_offset = i * 0.5  # Offset phase for each ball
                pulse_radius = abs(math.sin((pygame.time.get_ticks() + pulse_offset * 100) / (500.0 / (bpm + 0.1 / 60)))) * base_size + 1

                # Randomize the color slightly for each pulse
                randomized_color = (
                    max(0, min(255, trail_color[0] + random.randint(-30, 30))),
                    max(0, min(255, trail_color[1] + random.randint(-30, 30))),
                    max(0, min(255, trail_color[2] + random.randint(-30, 30)))
                )

                # Draw the small pulsing light ball
                pygame.draw.circle(screen, randomized_color, (screen_x, screen_y), int(pulse_radius), 0)

                # Add a subtle glowing outer ring
                if pulse_radius > 4:
                    glow_color = (
                        max(0, min(255, randomized_color[0] // 2)),
                        max(0, min(255, randomized_color[1] // 2)),
                        max(0, min(255, randomized_color[2] // 2))
                    )
                    pygame.draw.circle(screen, glow_color, (screen_x, screen_y), int(pulse_radius + 3), 1)

    else:
        # If no motion is detected, fill the screen with a calming glow or waves
        for _ in range(3):
            wave_x = random.randint(0, screen_width)
            wave_y = random.randint(0, screen_height)
            wave_radius = int(abs(math.sin(pygame.time.get_ticks() / 1000.0)) * 30) + 10
            pygame.draw.circle(screen, (128, 128, 255), (wave_x, wave_y), wave_radius, 1)

    # Update the display
    pygame.display.flip()
