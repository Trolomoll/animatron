import cv2
import platform

# Use DirectShow on Windows, default backend on Linux/Mac
def get_camera_backend():
    if platform.system() == 'Windows':
        return cv2.CAP_DSHOW
    else:
        return cv2.CAP_ANY  # Default backend for Linux/Mac

def find_cameras(max_cameras=10):
    """Find all available cameras"""
    backend = get_camera_backend()
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

def main():
    # Find available cameras
    print(f"Platform: {platform.system()}")
    print("Scanning for cameras...")
    cameras = find_cameras()
    
    if not cameras:
        print("No cameras found!")
        exit()
    
    print(f"\nFound {len(cameras)} camera(s) at indices: {cameras}")
    print("\nOpening all cameras in separate windows...")
    print("Press Q to quit\n")
    
    # Open all cameras simultaneously
    backend = get_camera_backend()
    caps = {}
    for idx in cameras:
        cap = cv2.VideoCapture(idx, backend)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        if cap.isOpened():
            caps[idx] = cap
            print(f"  Camera {idx}: Opened successfully")
        else:
            print(f"  Camera {idx}: Failed to open")
    
    if not caps:
        print("No cameras could be opened!")
        exit()

    while True:
        # Read and display from all cameras
        for idx, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                # Show camera index on frame
                cv2.putText(frame, f"Camera {idx}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f'Camera {idx} (Press Q to quit)', frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Quit on 'q'
        if key == ord('q'):
            break

    # Release all cameras
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()