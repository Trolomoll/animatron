import cv2
import numpy as np
import random
import time
import pygame

def spooky_filter_with_bpm(first_frame, bpm, last_beat_time, cap, screen_width, screen_height, screen):
    bpm = max(bpm, 1)  # Ensure BPM is at least 1 to avoid division by zero
    beat_interval = 60.0 / bpm

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 1)

    current_time = time.time()

    if first_frame is None:
        first_frame = gray

    if current_time - last_beat_time >= beat_interval:
        first_frame = gray
        last_beat_time = current_time

    if first_frame.shape != gray.shape:
        first_frame = gray.copy()

    delta_frame = cv2.absdiff(first_frame, gray)
    delta_frame_rgb = cv2.cvtColor(delta_frame, cv2.COLOR_GRAY2RGB)
    random_color = np.array([random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)], dtype=np.uint8)

    delta_frame_rgb_tinted = cv2.multiply(delta_frame_rgb*0.1, random_color)
    delta_frame_rgb_tinted = np.clip(delta_frame_rgb_tinted, 0, 255)
    delta_frame_rgb_tinted = cv2.convertScaleAbs(delta_frame_rgb_tinted, alpha=0.6)
    delta_frame_rgb_resized = cv2.resize(delta_frame_rgb_tinted, (screen_width, screen_height), interpolation=cv2.INTER_CUBIC)
    delta_frame_rgb_swapped = np.swapaxes(delta_frame_rgb_resized, 0, 1)

    surface = pygame.surfarray.make_surface(delta_frame_rgb_swapped)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    return first_frame, last_beat_time
