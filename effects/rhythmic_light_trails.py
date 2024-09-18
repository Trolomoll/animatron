import cv2
import pygame
import random
import math

def rhythmic_light_trails(cap, screen, screen_width, screen_height, black, bpm):
    ret, frame1 = cap.read()
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)
    ret, frame2 = cap.read()
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

    delta_frame = cv2.absdiff(frame1_gray, frame2_gray)
    thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set colors based on BPM (warmer colors for faster BPMs)
    if bpm < 80:
        trail_color = (0, 128, 255)  # Cool Blue
    elif bpm < 120:
        trail_color = (0, 255, 128)  # Green
    else:
        trail_color = (255, 128, 0)  # Warm Orange

    # Pulse size based on BPM
    pulse_size = 20 + (bpm / 2)

    # Screen trail effect background (slight fade out)
    overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 25))  # Create a translucent layer to give the fading effect
    screen.blit(overlay, (0, 0))

    if len(contours) > 0:
        # If motion is detected, generate glowing trails that pulse with BPM
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)

            # Create pulsing light trails
            pulse_radius = abs(math.sin(pygame.time.get_ticks() / (500.0 / (bpm / 60)))) * pulse_size

            # Randomize the color slightly for each pulse and ensure valid RGB range (0-255)
            randomized_color = (
                max(0, min(255, trail_color[0] + random.randint(-30, 30))),
                max(0, min(255, trail_color[1] + random.randint(-30, 30))),
                max(0, min(255, trail_color[2] + random.randint(-30, 30)))
            )

            # Draw the pulsing light trail
            pygame.draw.circle(screen, randomized_color, (int(x * screen_width / frame1.shape[1]), int(y * screen_height / frame1.shape[0])), int(pulse_radius), 0)

            # Add a glowing outer ring
            pygame.draw.circle(screen, randomized_color, (int(x * screen_width / frame1.shape[1]), int(y * screen_height / frame1.shape[0])), int(pulse_radius + 10), 2)

    else:
        # If no motion is detected, fill the screen with a calming glow or waves
        for _ in range(5):
            wave_x = random.randint(0, screen_width)
            wave_y = random.randint(0, screen_height)
            wave_radius = int(abs(math.sin(pygame.time.get_ticks() / 1000.0)) * 50) + 20
            pygame.draw.circle(screen, (128, 128, 255), (wave_x, wave_y), wave_radius, 1)

    # Update the display
    pygame.display.flip()
