import cv2
import numpy as np
import pygame
import time
import pyaudio
import librosa
import random

# Initialize webcam and Pygame
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
black = (0, 0, 0)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# Settings for the microphone input
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate (44.1 kHz)

# Initialize PyAudio
p = pyaudio.PyAudio()

first_frame = None
running = True


def get_bpm_from_microphone(seconds=5):
    """Capture audio from the microphone and estimate the BPM."""
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"Recording for {seconds} seconds...")

    frames = []

    # Capture audio data for the specified duration
    for _ in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    stream.stop_stream()
    stream.close()

    # Convert the frames to a 1D array
    audio_data = np.hstack(frames)

    # Use librosa to estimate the BPM from the audio data
    tempo, _ = librosa.beat.beat_track(y=audio_data.astype(float), sr=RATE)
    
    return tempo


def colorful_circles():
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

    screen.fill(black)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        color = random.choice(colors)
        # Draw circle with adjusted position to match the Pygame canvas size
        pygame.draw.circle(screen, color, (int(x * screen_width / frame1.shape[1]), 
                                           int(y * screen_height / frame1.shape[0])), 
                           random.randint(10, 50))

    pygame.display.flip()



def spooky_filter_with_bpm(first_frame, bpm, last_beat_time):
    # Calculate the time interval between beats based on BPM
    beat_interval = 60.0 / bpm  # Seconds per beat

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    current_time = time.time()

    # Initialize first_frame if it's None
    if first_frame is None:
        first_frame = gray

    # Update first_frame based on BPM
    if current_time - last_beat_time >= beat_interval:
        first_frame = gray
        last_beat_time = current_time

    # Ensure first_frame and gray have the same size and number of channels
    if first_frame.shape != gray.shape:
        first_frame = gray.copy()

    delta_frame = cv2.absdiff(first_frame, gray)

    # Convert grayscale to 3-channel (RGB) image for Pygame compatibility
    delta_frame_rgb = cv2.cvtColor(delta_frame, cv2.COLOR_GRAY2RGB)

    # Generate random RGB color values
    random_color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)

    # Apply random color tint by multiplying each channel by the random color
    delta_frame_rgb_tinted = cv2.multiply(delta_frame_rgb, random_color)

    # Resize the image to match Pygame screen dimensions
    delta_frame_rgb_resized = cv2.resize(delta_frame_rgb_tinted, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    # Swap the axes of the image (effectively rotating it)
    delta_frame_rgb_swapped = np.swapaxes(delta_frame_rgb_resized, 0, 1)

    # Create a Pygame surface from the resized image
    surface = pygame.surfarray.make_surface(delta_frame_rgb_swapped)

    # Blit the surface onto the Pygame screen
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    return first_frame, last_beat_time  # Return the updated first_frame and last_beat_time



def edge_detection():
    """Apply Canny edge detection to the webcam feed and display it."""
    ret, frame = cap.read()
    if not ret:
        return

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Convert the edges to RGB format so it can be displayed with Pygame
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Resize the edges image to match Pygame screen dimensions
    edges_resized = cv2.resize(edges_rgb, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    # Swap the axes of the image to match Pygame's (width, height) format
    edges_swapped = np.swapaxes(edges_resized, 0, 1)

    # Create a Pygame surface from the resized image
    surface = pygame.surfarray.make_surface(edges_swapped)

    # Blit the surface onto the Pygame screen
    screen.blit(surface, (0, 0))
    pygame.display.flip()


def ascii_webcam_effect():
    """Apply ASCII effect to the webcam feed and display it using Pygame."""
    chars = np.asarray(list(' .,:irs?@9B&#'))

    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        return

    # Resize the frame to a smaller size for ASCII conversion
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    new_width = 100
    new_height = int(aspect_ratio * new_width * 0.55)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Normalize the grayscale image to a range between 0 and the number of ASCII characters
    normalized_gray = (gray / 255) * (chars.size - 1)

    # Convert the grayscale values to ASCII characters
    ascii_image = chars[normalized_gray.astype(int)]

    # Join the ASCII characters to form the image
    ascii_image_str = "\n".join("".join(row) for row in ascii_image)

    # Display the ASCII image on the Pygame screen
    font = pygame.font.SysFont('Courier', 12)
    screen.fill(black)
    
    # Render the ASCII string character by character
    for i, line in enumerate(ascii_image_str.split("\n")):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (0, i * 12))  # 12 is the font size for vertical spacing

    pygame.display.flip()

def circle_grid_effect(bpm, previous_frame=None):
    """Apply the circle grid effect to the webcam feed with color tint and afterimage effect."""
    cell_width, cell_height = 12, 12
    new_width, new_height = int(screen_width / cell_width), int(screen_height / cell_height)

    # Define color intensity adjustment factor based on BPM
    color_intensity_factor = np.interp(bpm, [60, 180], [0.5, 1.5])  # Adjust between 0.5 and 1.5 based on BPM

    # Generate a random or BPM-dependent RGB tint
    tint_color = np.array([random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)], dtype=np.uint8)

    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        return previous_frame  # Return previous frame if no new frame is captured

    # Create a black window for drawing the current frame
    black_window = np.zeros((screen_height, screen_width, 3), np.uint8)

    # Resize the frame to fit the grid
    small_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    for i in range(new_height):
        for j in range(new_width):
            color = small_image[i, j]
            
            # Apply color intensity adjustment
            B = int(color[0] * color_intensity_factor)
            G = int(color[1] * color_intensity_factor)
            R = int(color[2] * color_intensity_factor)
            
            # Apply the tint color by multiplying the current pixel's color by the tint
            B = int(B * tint_color[0] / 255)
            G = int(G * tint_color[1] / 255)
            R = int(R * tint_color[2] / 255)

            # Ensure the values are within valid RGB range (0-255)
            B, G, R = [int(np.clip(B, 0, 255)), int(np.clip(G, 0, 255)), int(np.clip(R, 0, 255))]

            coord = (j * cell_width + cell_width, i * cell_height)

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
    resized_frame = cv2.resize(black_window_rgb, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
    swapped_frame = np.swapaxes(resized_frame, 0, 1)

    # Create a Pygame surface from the swapped frame
    surface = pygame.surfarray.make_surface(swapped_frame)

    # Blit the surface onto the Pygame screen
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    return previous_frame  # Return the updated previous frame





def switch_effects(running, bpm):
    """Switch between effects every 10 seconds."""
    start_time = time.time()
    first_frame = None  # Initialize first_frame for spooky filter
    last_beat_time = time.time()  # Track time for BPM-based switching
    effect_switch_time = 10  # Time interval to switch effects (in seconds)
    current_effect = 0  # 0 for colorful circles, 1 for spooky filter, 2 for edge detection, 3 for ASCII, 4 for circle grid
    previous_frame = None  # Initialize for afterimage effect in circle grid

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = time.time()

        # Check if it's time to switch effects
        if current_time - start_time > effect_switch_time:
            current_effect = (current_effect + 1) % 5  # Cycle through 0, 1, 2, 3, and 4 for the five effects
            start_time = current_time  # Reset the effect switch timer

        # Apply the selected effect
        if current_effect == 0:
            previous_frame = circle_grid_effect(bpm, previous_frame)  # Pass previous_frame for afterimage
        elif current_effect == 1:
            colorful_circles()
        elif current_effect == 2:
            edge_detection()
        elif current_effect == 3:
            ascii_webcam_effect()
        # elif current_effect == x:
        #     elon_musk()
        # elif current_effect == x:
        #     random_environment_info()
        else:
            first_frame, last_beat_time = spooky_filter_with_bpm(first_frame, bpm, last_beat_time)

        # Update display and frame rate
        pygame.display.flip()



# Get BPM from microphone
bpm = get_bpm_from_microphone(5)
print(f"Estimated BPM: {bpm}")

# Run the effect switcher with BPM-based spooky filter
switch_effects(running, bpm)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
