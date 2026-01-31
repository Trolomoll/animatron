import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import pygame
import time

# Initialize the FaceDetector object
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Function to compute smiley color based on BPM and beat timing
def get_smiley_color(bpm, last_beat_time):
    current_time = time.time()
    beat_interval = 60.0 / bpm
    time_since_last_beat = (current_time - last_beat_time) % beat_interval
    beat_progress = time_since_last_beat / beat_interval  # Between 0 and 1

    # Interpolate color between yellow (0) and red (1)
    start_color = np.array([0, 255, 255])  # Yellow
    end_color = np.array([0, 0, 255])  # Red
    color = (1 - beat_progress) * start_color + beat_progress * end_color

    # Ensure color is a tuple of integers (required by OpenCV)
    return tuple(map(int, color))

# Function to draw a smiley face inside a bounding box with dynamic color
def _expanded_face_geometry(img, x, y, w, h):
    img_h, img_w = img.shape[:2]

    # Expand bounding box to cover full head (including hair)
    expand_top = int(h * 0.4)
    expand_sides = int(w * 0.2)

    new_x = max(0, x - expand_sides)
    new_y = max(0, y - expand_top)
    new_w = min(img_w - new_x, w + expand_sides * 2)
    new_h = min(img_h - new_y, h + expand_top)

    center_x = new_x + new_w // 2
    center_y = new_y + new_h // 2
    radius = max(1, min(new_w, new_h) // 2)

    return {
        "x": new_x,
        "y": new_y,
        "w": new_w,
        "h": new_h,
        "cx": center_x,
        "cy": center_y,
        "r": radius,
    }


def draw_smiley(img, x, y, w, h, smiley_color):
    g = _expanded_face_geometry(img, x, y, w, h)
    center_x, center_y, new_w, new_h = g["cx"], g["cy"], g["w"], g["h"]

    # Draw the face (head) with the dynamic smiley color
    cv2.circle(img, (center_x, center_y), g["r"], smiley_color, -1)

    # Draw the eyes (positioned relative to the expanded head)
    eye_radius = min(new_w, new_h) // 10
    eye_offset_x = new_w // 4
    eye_offset_y = new_h // 8  # Adjusted for larger head
    cv2.circle(img, (center_x - eye_offset_x, center_y - eye_offset_y), eye_radius, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (center_x + eye_offset_x, center_y - eye_offset_y), eye_radius, (0, 0, 0), -1)  # Right eye

    # Draw the smile (arc/ellipse)
    smile_thickness = 3
    smile_radius_x = new_w // 5
    smile_radius_y = new_h // 10
    smile_start_angle = 20
    smile_end_angle = 160
    cv2.ellipse(img, (center_x, center_y + new_h // 6), (smile_radius_x, smile_radius_y),
                0, smile_start_angle, smile_end_angle, (0, 0, 0), smile_thickness)


def draw_confused(img, x, y, w, h, smiley_color):
    g = _expanded_face_geometry(img, x, y, w, h)
    center_x, center_y, new_w, new_h = g["cx"], g["cy"], g["w"], g["h"]

    # Head
    cv2.circle(img, (center_x, center_y), g["r"], smiley_color, -1)

    # Eyes
    eye_radius = max(1, min(new_w, new_h) // 10)
    eye_offset_x = new_w // 4
    eye_offset_y = new_h // 8
    cv2.circle(img, (center_x - eye_offset_x, center_y - eye_offset_y), eye_radius, (0, 0, 0), -1)
    cv2.circle(img, (center_x + eye_offset_x, center_y - eye_offset_y), eye_radius, (0, 0, 0), -1)

    # Confused eyebrows (slanted)
    brow_y = center_y - eye_offset_y - eye_radius * 2
    brow_len = max(6, new_w // 8)
    brow_thickness = 3
    cv2.line(
        img,
        (center_x - eye_offset_x - brow_len // 2, brow_y + 6),
        (center_x - eye_offset_x + brow_len // 2, brow_y - 2),
        (0, 0, 0),
        brow_thickness,
    )
    cv2.line(
        img,
        (center_x + eye_offset_x - brow_len // 2, brow_y - 2),
        (center_x + eye_offset_x + brow_len // 2, brow_y + 6),
        (0, 0, 0),
        brow_thickness,
    )

    # Confused mouth: a little wavy squiggle
    mouth_y = center_y + new_h // 6
    mouth_w = max(12, new_w // 3)
    mouth_h = max(4, new_h // 18)
    x0 = center_x - mouth_w // 2
    pts = []
    steps = 7
    for i in range(steps):
        t = i / (steps - 1)
        x_i = int(x0 + t * mouth_w)
        y_i = int(mouth_y + np.sin(t * np.pi * 2.0) * mouth_h)
        pts.append([x_i, y_i])
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 0), thickness=3)

# Face detection effect function with smileys that change color based on BPM
def face_detection_effect_with_bpm(cap, screen, screen_width, screen_height, bpm, last_beat_time):
    # Read the current frame from the webcam
    success, img = cap.read()
    
    if not success:
        return last_beat_time

    # Detect faces in the image
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected and draw bounding boxes and smileys
    if bboxs:
        smiley_color = get_smiley_color(bpm, last_beat_time)  # Get the dynamic color based on BPM

        faces = []
        masks = []
        bounds = []
        img_h, img_w = img.shape[:2]

        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            g = _expanded_face_geometry(img, x, y, w, h)
            faces.append((x, y, w, h, g))

            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.circle(mask, (g["cx"], g["cy"]), g["r"], 255, -1)
            masks.append(mask)

            bounds.append((g["cx"] - g["r"], g["cy"] - g["r"], g["cx"] + g["r"], g["cy"] + g["r"]))

        collided = [False] * len(faces)
        for i in range(len(faces)):
            x1a, y1a, x2a, y2a = bounds[i]
            for j in range(i + 1, len(faces)):
                x1b, y1b, x2b, y2b = bounds[j]

                ox1 = max(0, max(x1a, x1b))
                oy1 = max(0, max(y1a, y1b))
                ox2 = min(img_w, min(x2a, x2b))
                oy2 = min(img_h, min(y2a, y2b))

                if ox2 <= ox1 or oy2 <= oy1:
                    continue

                roi_a = masks[i][oy1:oy2, ox1:ox2]
                roi_b = masks[j][oy1:oy2, ox1:ox2]
                if np.any(cv2.bitwise_and(roi_a, roi_b)):
                    collided[i] = True
                    collided[j] = True

        for idx, (x, y, w, h, _g) in enumerate(faces):
            if collided[idx]:
                draw_confused(img, x, y, w, h, smiley_color)
            else:
                draw_smiley(img, x, y, w, h, smiley_color)

    # Resize the frame to fit the Pygame screen dimensions
    img_resized = cv2.resize(img, (screen_width, screen_height))

    # Convert the OpenCV image (BGR) to the Pygame surface (RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = np.swapaxes(img_rgb, 0, 1)

    face_surface = pygame.surfarray.make_surface(img_rgb)

    # Display the face detection result in Pygame
    screen.blit(pygame.transform.scale(face_surface, (screen_width, screen_height)), (0, 0))

    return last_beat_time
