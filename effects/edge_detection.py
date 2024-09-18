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
