import time

import cv2
import numpy as np
import pygame


def _make_rainbow_bgr(width: int, height: int, phase: float) -> np.ndarray:
    """Create a vivid rainbow image (BGR) using an HSV gradient."""
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]

    # Diagonal gradient + time phase
    hue = (x + y + phase) % 180.0

    hsv = np.empty((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = hue.astype(np.uint8)  # 0..179
    hsv[..., 1] = 255
    hsv[..., 2] = 255

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def motion_rainbow_effect(
    state,
    cap,
    screen,
    screen_width: int,
    screen_height: int,
    motion_threshold: int = 20,
):
    """Overlay funny moving rainbows only where motion is detected.

    `state` is an internal dict that persists between frames.
    """
    if state is None:
        state = {
            "prev_small_gray": None,
            "phase": 0.0,
            "last_t": time.time(),
        }

    ret, frame = cap.read()
    if not ret:
        return state

    # Resize early so our output matches the display.
    frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    # Show the camera as a negative image.
    frame_base = cv2.bitwise_not(frame)

    # Small grayscale for motion detection (faster + less noisy).
    small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.GaussianBlur(small_gray, (7, 7), 0)

    prev = state["prev_small_gray"]
    if prev is None or prev.shape != small_gray.shape:
        state["prev_small_gray"] = small_gray

        # Show negative camera on the first frame.
        rgb = cv2.cvtColor(frame_base, cv2.COLOR_BGR2RGB)
        swapped = np.swapaxes(rgb, 0, 1)
        surface = pygame.surfarray.make_surface(swapped)
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        return state

    diff = cv2.absdiff(prev, small_gray)
    _, motion_mask_small = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)
    motion_mask_small = cv2.dilate(motion_mask_small, None, iterations=2)

    motion_ratio = float(np.count_nonzero(motion_mask_small)) / float(motion_mask_small.size)
    motion_strength = max(0.0, min(1.0, motion_ratio * 8.0))

    now = time.time()
    dt = max(0.0, now - float(state["last_t"]))

    # Keep the rainbow mostly idle unless there is motion.
    base_speed = 15.0
    motion_speed = 500.0 * motion_strength
    state["phase"] = float(state["phase"]) + dt * (base_speed + motion_speed)
    state["last_t"] = now

    # Create a smaller rainbow buffer and scale up for performance.
    overlay_small = _make_rainbow_bgr(360, 200, state["phase"])

    # Make it feel "alive" when you move.
    dx = int(np.sin(state["phase"] / 55.0) * 40.0 * motion_strength)
    dy = int(np.cos(state["phase"] / 65.0) * 25.0 * motion_strength)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    overlay_small = cv2.warpAffine(
        overlay_small,
        m,
        (overlay_small.shape[1], overlay_small.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )

    overlay = cv2.resize(overlay_small, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    motion_mask = cv2.resize(
        motion_mask_small, (screen_width, screen_height), interpolation=cv2.INTER_NEAREST
    )
    motion_mask = cv2.GaussianBlur(motion_mask, (31, 31), 0)

    # Only tint where there is motion (with a small baseline so it feels playful).
    alpha = (motion_mask.astype(np.float32) / 255.0) * (0.05 + 0.85 * motion_strength)
    alpha3 = alpha[:, :, None]

    out = (
        frame_base.astype(np.float32) * (1.0 - alpha3) + overlay.astype(np.float32) * alpha3
    ).astype(np.uint8)

    # Update motion baseline for next frame.
    state["prev_small_gray"] = small_gray

    # Display via pygame (expects RGB + swapped axes)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out_swapped = np.swapaxes(out_rgb, 0, 1)
    surface = pygame.surfarray.make_surface(out_swapped)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    return state
