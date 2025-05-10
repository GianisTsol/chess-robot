import cv2
import numpy as np
import sys
import os

def nothing(x):
    pass

def select_camera():
    """Try to find a valid camera index."""
    for i in range(10):  # Try first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found camera at index {i}")
            cap.release()
            return i
    return 0  # Default to 0 if no camera found

def calibrate_hsv():
    """Interactive HSV calibration tool."""
    
    # Try to find a working camera
    camera_index = select_camera()
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera at index {camera_index}")
        print("Please enter a valid camera index:")
        camera_index = int(input())
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Failed to open camera. Exiting.")
            return

    print(f"Using camera index: {camera_index}")
    print("Press 's' to save current HSV values")
    print("Press 'q' to quit")
    
    # Create windows
    cv2.namedWindow('Original')
    cv2.namedWindow('White Mask')
    cv2.namedWindow('HSV Controls')

    # Create trackbars for HSV range control
    cv2.createTrackbar('Hue Min', 'HSV Controls', 0, 179, nothing)
    cv2.createTrackbar('Hue Max', 'HSV Controls', 179, 179, nothing)
    cv2.createTrackbar('Sat Min', 'HSV Controls', 0, 255, nothing)
    cv2.createTrackbar('Sat Max', 'HSV Controls', 255, 255, nothing)
    cv2.createTrackbar('Val Min', 'HSV Controls', 0, 255, nothing)
    cv2.createTrackbar('Val Max', 'HSV Controls', 255, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Flip frame horizontally for more intuitive movement
        frame = cv2.flip(frame, 1)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get current trackbar positions
        h_min = cv2.getTrackbarPos('Hue Min', 'HSV Controls')
        h_max = cv2.getTrackbarPos('Hue Max', 'HSV Controls')
        s_min = cv2.getTrackbarPos('Sat Min', 'HSV Controls')
        s_max = cv2.getTrackbarPos('Sat Max', 'HSV Controls')
        v_min = cv2.getTrackbarPos('Val Min', 'HSV Controls')
        v_max = cv2.getTrackbarPos('Val Max', 'HSV Controls')
        
        # Create lower and upper bounds for the white color
        lower_white = np.array([h_min, s_min, v_min])
        upper_white = np.array([h_max, s_max, v_max])
        
        # Create mask and result
        mask = cv2.inRange(hsv, lower_white, upper_white)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Show images
        cv2.imshow('Original', frame)
        cv2.imshow('White Mask', mask)
        cv2.imshow('Result', result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"Current HSV values - Min: {lower_white}, Max: {upper_white}")
            # Save values to file
            with open('hsv_values.txt', 'w') as f:
                f.write(f"{h_min},{s_min},{v_min}\n")
                f.write(f"{h_max},{s_max},{v_max}\n")
            print("HSV values saved to hsv_values.txt")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_hsv()