"""
This script detects ArUco markers and estimates their poses using camera calibration values.
It displays the original image and the image with detected markers and their poses.
It requires OpenCV 4.7 or newer, including OpenCV 5, and uses ArucoDetector and cv2.solvePnP
(the older aruco.detectMarkers and aruco.estimatePoseSingleMarkers functions do not exist in OpenCV 5).

@author: ioannisgeorgilas
"""

import struct
import cv2
import cv2.aruco as aruco
import logging
import numpy as np
import time # We will use this to ensure a steady processing rate
# First we import our libraries
import socket   # This library will allow you to communicate over the network
import logging

#set the logging level to INFO
logging.basicConfig(level=logging.INFO)

# This is the IP address of the machine that the data will be send to
#TCP_IP = "138.38.229.102"
#UDP_IP = "138.38.229.102" # big pi
UDP_IP = "138.38.226.156" # small pi

# This is the REMOTE port of the Server that we are sending the data to
#TCP_PORT = 25000

# This is the REMOTE port the machine will reply on (on that machine this is the value for the LOCAL port)
UDP_PORT = 50001
# Create the socket for the TCP communication
#s = socket.socket(socket.AF_INET,        # Family of addresses, in this case IP type 
                     #socket.SOCK_STREAM)    # What protocol to use, in this case TCP (streamed communications)
#logging.info('Socket successfully created')


# Create the socket for the UDP communication
sock = socket.socket(socket.AF_INET,    # Family of addresses, in this case IP type 
                     socket.SOCK_DGRAM) # What protocol to use, in this case UDP (datagram)

# Establish the connection with the remote Server using their IP and the port                                       
#s.connect((TCP_IP, TCP_PORT))
#logging.info('Connected')

# Load the camera calibration values
camera_calibration = np.load('Sample_Calibration.npz')
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef']# distortion coefficients from the camera
logging.info("Loaded camera calibration from Sample_Calibration.npz")

#define lazer and motor variables
LAZER = 0
MOTOR = 0

# Define the ArUco dictionary and parameters
marker_size = 40 # the real size of the printed marker in mm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# The 3D position (in mm) of the 4 corners of a marker, in the marker's own coordinate system (origin in the centre).
# The order is the same as the corners returned by the detector: top-left, top-right, bottom-right, bottom-left
half = marker_size / 2
marker_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

# Define a processing rate in milliseconds (e.g., 0.25 means 4 frames per second).
processing_period = 0.1

# ask user for aruco code to be hit
user_input = input("Please enter an aruco code you would like to hit: ")

# Create two OpenCV named windows
cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Gray", cv2.WINDOW_AUTOSIZE)

# Position the windows next to each other
cv2.moveWindow("Gray", 640, 100)
cv2.moveWindow("Frame", 0, 100)

# Start capturing video
cap = cv2.VideoCapture(1) # 0 for laptop camera, 1 for usb camera
if not cap.isOpened():
    logging.error("Could not open camera 1")
    raise SystemExit(1)
logging.info("Camera 1 opened")

# Set the starting time
start_time = time.time()
fps = 0
last_logged_marker_ids = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        logging.error("Could not read a frame from camera 1; exiting")
        break

    frame_height, frame_width = frame.shape[:2]
    fx, fy = CM[0, 0], CM[1, 1]
    cx, cy = CM[0, 2], CM[1, 2]

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)

    # Detect markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    marker_ids = tuple(sorted(int(marker_id) for marker_id in ids.flatten())) if ids is not None else ()
    if marker_ids != last_logged_marker_ids:
        if marker_ids:
            logging.info("Detected ArUco marker IDs: %s", marker_ids)
        elif last_logged_marker_ids:
            logging.info("No ArUco markers detected")
        last_logged_marker_ids = marker_ids

    #define variables
    tx = 0.0
    ty = 0.0
    tz = 0.0
    marker_id = -1
    alpha = 0.0
    marker_tx = 0.0
    marker_ty = 0.0
    marker_tz = 0.0
    marker_alpha = 0.0
    tx_simulink = 0
    alpha_simulink = 0

    # If markers are detected
    # (when nothing is detected ids is None in OpenCV 4 but an empty array in OpenCV 5, so we check the corners)
    if len(corners) > 0:
        # Draw detected markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate the pose of each marker from its 4 detected corners
        closest_abs_x = float("inf")
        for i in range(len(corners)):
            ok, rvec, tvec = cv2.solvePnP(marker_points, corners[i], CM, dist_coef, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                marker_tx, marker_ty, marker_tz = tvec.ravel()
                marker_tx += (cx - frame_width / 2) * marker_tz / fx
                marker_ty += (cy - frame_height / 2) * marker_tz / fy
                marker_alpha = np.arctan(marker_ty / 590.0) * 180 / np.pi
                detected_marker_id = ids[i].item()
                print(f"Marker ID {detected_marker_id}: frame-centred X = {marker_tx:.1f}, Y = {marker_ty:.1f}, Z = {marker_tz:.1f} mm: angle = {marker_alpha:.1f} degrees")
                # Draw axis for each marker (axis length 10 mm)
                frame = cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 10)

                if abs(marker_tx) < closest_abs_x:
                    tx, ty, tz = marker_tx, marker_ty, marker_tz
                    alpha = marker_alpha
                    tx_simulink = tx*10
                    alpha_simulink = alpha*10
                    marker_id = detected_marker_id
                    closest_abs_x = abs(marker_tx)
        #test to see if different marker id's can output different things
        if marker_id == 43:
            LAZER = 1
            MOTOR = 0
        elif marker_id == 35:
            MOTOR = 1
            LAZER = 0
        else:
            LAZER = 0
            MOTOR = 0

        print(f"Lazer {LAZER}, Motor {MOTOR}")

    # Add the frame rate to the image
    cv2.putText(frame, f"CAMERA FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"PROCESSING FPS: {1/processing_period:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break

    # Ensure a steady processing rate
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    if elapsed_time < processing_period:
        time.sleep(processing_period - elapsed_time)
    start_time = time.time()

    # Send tx, alpha, marker_id and user_input as four int16 to keep the sign (2 bytes each (8 bytes total)).
    data = struct.pack("<4h", int(tx_simulink), int(alpha_simulink), int(marker_id), int(user_input))
    sock.sendto(data, (UDP_IP, UDP_PORT))

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
sock.close()
logging.info("UDP socket closed")
logging.info("Camera capture stopped")