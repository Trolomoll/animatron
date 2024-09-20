import cv2
import numpy as np
import pygame
import time
import threading
import pyaudio
import librosa

# Import the separated effect files
from effects.matrix_effect import matrix_effect
from effects.smiley_effect import face_detection_effect
from effects.spooky_filter import spooky_filter_with_bpm
from effects.colorful_circles import colorful_circles
from effects.edge_detection import edge_detection, edge_detection_with_afterimage, edge_detection_with_pulse_and_afterimage
from effects.ascii_webcam_effect import ascii_webcam_effect
from effects.circle_grid_effect import circle_grid_effect
from effects.rhythmic_light_trails import rhythmic_light_trails
from effects.edge_detection_two import edge_detection_two

# Initialize webcam and Pygame
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

pygame.init()
""" screen = pygame.display.set_mode((800, 600))  # Fullscreen mode
screen_width, screen_height = pygame.display.get_surface().get_size() """

screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE | pygame.FULLSCREEN)  # Fullscreen mode
screen_width, screen_height = pygame.display.get_surface().get_size()

black = (0, 0, 0)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# Settings for the microphone input
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate (44.1 kHz)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Initialize BPM and threading lock
bpm = 120  # Initial BPM
bpm_lock = threading.Lock()
running = True  # Control variable for threads
first_frame = None

def estimate_bpm(audio_data):
    """Estimate the BPM from audio data, ensuring n_fft is not larger than the input length."""
    if len(audio_data) < 2048:
        n_fft = len(audio_data)  # Adjust FFT size to fit the audio data length
    else:
        n_fft = 2048  # Use the default FFT size for larger audio buffers
    
    # tempo, _ = librosa.beat.beat_track(y=audio_data.astype(float), sr=RATE, n_fft=n_fft)
    tempo, _ = librosa.beat.beat_track(y=audio_data.astype(float), sr=RATE)
    return tempo

def run_bpm_estimation():
    """Continuously capture audio and update BPM estimation."""
    global bpm, running
    # Set up PyAudio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    audio_buffer = np.array([], dtype=np.int16)
    start_time = time.time()

    while running:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.concatenate((audio_buffer, audio_data))
        except Exception as e:
            print(f"Audio stream error: {e}")
            continue

        current_time = time.time()
        if current_time - start_time >= 5:  # Update BPM every 5 seconds
            # Ensure we are using enough audio data for BPM estimation
            if len(audio_buffer) >= CHUNK * 5:
                temp_bpm = estimate_bpm(audio_buffer)
                with bpm_lock:
                    bpm = temp_bpm
            else:
                print("Not enough audio data for BPM estimation.")
                
            # Reset buffer and timer
            audio_buffer = np.array([], dtype=np.int16)
            start_time = current_time

    stream.stop_stream()
    stream.close()


def display_bpm_on_screen(current_bpm):
    """Display the current BPM in the upper-right corner of the screen."""
    font = pygame.font.SysFont("Arial", 30)
    bpm_text = font.render(f"BPM: {int(current_bpm)}", True, (255, 255, 255))
    screen.blit(bpm_text, (screen_width - bpm_text.get_width() - 20, 20))

def switch_effects():
    """Switch between effects every 10 seconds and display BPM."""
    global running  # Access the global running variable
    start_time = time.time()
    first_frame = None  # Initialize first_frame for spooky filter
    last_beat_time = time.time()  # Track time for BPM-based switching
    effect_switch_time = 20  # Time interval to switch effects (in seconds)
    current_effect = 0  # 0 for circle grid, 1 for colorful circles, etc.
    previous_frame = None  # Initialize for afterimage effect in circle grid

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Add a key press listener for the 'X' key to stop the program
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:  # Pygame's constant for the 'X' key
                    running = False
                elif event.key == pygame.K_n:  # Pygame's constant for the 'N' key to switch effects
                    current_effect = (current_effect + 1) % 8  # Immediately switch to the next effect
                    start_time = time.time()  # Reset the timer so it doesn't immediately switch again
                elif event.key == pygame.K_ESCAPE:  # Pygame's constant for the 'ESC' key
                    running = False
        current_time = time.time()

        # Check if it's time to switch effects based on the timer
        if current_time - start_time > effect_switch_time:
            current_effect = (current_effect + 1) % 10  # Cycle through effects
            start_time = current_time  # Reset the effect switch timer

        # Safely read the current BPM
        with bpm_lock:
            current_bpm = bpm

        if current_bpm == 0:
            current_bpm = 60

        # Apply the selected effect
        if current_effect == 0:
            first_frame, last_beat_time = spooky_filter_with_bpm(first_frame, current_bpm, last_beat_time, cap, screen_width, screen_height, screen)
        elif current_effect == 1:
            face_detection_effect(cap, screen, screen_width, screen_height)
        elif current_effect == 7:
            matrix_effect(cap, screen, screen_width, screen_height)
        # elif current_effect == 6:
        #     colorful_circles(cap, screen, screen_width, screen_height, black)
        elif current_effect == 2:
            edge_detection(cap, screen, screen_width, screen_height)
        elif current_effect == 3:
            edge_detection_with_pulse_and_afterimage(cap, screen, screen_width, screen_height, current_bpm)
        elif current_effect == 4:
            rhythmic_light_trails(cap, screen, screen_width, screen_height, black, bpm)
        elif current_effect == 5:
            last_beat_time = ascii_webcam_effect(current_bpm, last_beat_time, cap, screen, screen_width, screen_height, black)
        # elif current_effect == 6:
        #     last_beat_time = edge_detection_with_afterimage(cap, screen, screen_width, screen_height)
        else:
            first_frame, last_beat_time = spooky_filter_with_bpm(first_frame, current_bpm, last_beat_time, cap, screen_width, screen_height, screen)

        # Display the current BPM in the upper-right corner
        display_bpm_on_screen(current_bpm)

        # Update display
        pygame.display.flip()

    # Stop the Pygame and webcam processes after the loop ends
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

# Start BPM estimation in a separate thread
bpm_thread = threading.Thread(target=run_bpm_estimation)
bpm_thread.start()

# Run the effect switcher
switch_effects()

# Cleanup
running = False  # Stop the BPM estimation thread
bpm_thread.join()
cap.release()
cv2.destroyAllWindows()
pygame.quit()
