from function import *
from time import sleep
import os
import cv2
import numpy as np

# Create directories for storing data
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Set up MediaPipe Hands model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Loop through actions
    for action in actions:
        # Loop through sequences (videos)
        for sequence in range(no_sequences):
            # Loop through frames in the video (sequence length)
            for frame_num in range(sequence_length):

                # Read the image frame
                frame = cv2.imread(f'Image/{action}/{sequence}.png')

                # Check if the image is loaded properly
                if frame is None:
                    print(f"Error: Failed to load image 'Image/{action}/{sequence}.png'")
                    continue  # Skip to the next frame if image is not loaded

                # Make detections using MediaPipe
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks on the image
                draw_styled_landmarks(image, results)
                
                # Apply wait logic and display feedback on the first frame
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)  # Wait 200 ms
                else: 
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints to .npy file
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break the loop gracefully if 'q' is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
