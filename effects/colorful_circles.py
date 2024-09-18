import cv2
import pygame
import random
import math

def colorful_circles(cap, screen, screen_width, screen_height, black):
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

    # Pastel, nature-inspired color palette
    pastel_colors = [
        (204, 229, 255),  # Light Blue
        (255, 204, 204),  # Soft Pink
        (204, 255, 204),  # Light Green
        (255, 229, 204),  # Soft Orange
        (255, 255, 204),  # Pale Yellow
        (229, 204, 255),  # Light Purple
        (204, 255, 229),  # Mint
    ]

    # Background animation (moving gradient)
    for i in range(0, screen_width, 20):
        for j in range(0, screen_height, 20):
            color = (random.randint(180, 220), random.randint(180, 220), random.randint(200, 255))
            pygame.draw.rect(screen, color, pygame.Rect(i, j, 20, 20))

    if len(contours) > 0:
        # Enhance motion detection effect with pastel bursts and large, fast-moving circles
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)

            # Pastel-colored bursts for motion
            base_radius = random.randint(30, 70)
            for _ in range(5):
                radius = abs(math.sin(pygame.time.get_ticks() / 500.0)) * base_radius + 20
                flash_color = random.choice(pastel_colors)
                pygame.draw.circle(screen, flash_color, (int(x * screen_width / frame1.shape[1]), int(y * screen_height / frame1.shape[0])), int(radius), 3)

            # Add streaks that shoot outward from the motion area
            streak_length = random.randint(50, 100)
            streak_angle = random.uniform(0, 2 * math.pi)
            end_x = int(x * screen_width / frame1.shape[1] + streak_length * math.cos(streak_angle))
            end_y = int(y * screen_height / frame1.shape[0] + streak_length * math.sin(streak_angle))
            pygame.draw.line(screen, flash_color, (int(x * screen_width / frame1.shape[1]), int(y * screen_height / frame1.shape[0])), (end_x, end_y), 3)

    else:
        # If no motion is detected, draw random pastel shapes floating on the screen
        for _ in range(10):
            shape_color = random.choice(pastel_colors)
            shape_x = random.randint(0, screen_width)
            shape_y = random.randint(0, screen_height)
            size = random.randint(20, 50)
            shape_type = random.choice(["circle", "rect"])

            if shape_type == "circle":
                pygame.draw.circle(screen, shape_color, (shape_x, shape_y), size)
            else:
                pygame.draw.rect(screen, shape_color, pygame.Rect(shape_x, shape_y, size, size))

    # Update the display
    pygame.display.flip()
