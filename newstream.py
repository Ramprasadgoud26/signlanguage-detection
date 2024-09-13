import streamlit as st
import numpy as np
import cv2
from function import mediapipe_detection, extract_keypoints, actions
from keras.models import model_from_json
import mediapipe as mp

# Initialize session state
if 'model' not in st.session_state:
    # Load the model only once
    json_file = open("model.json", "r")
    model_json = json_file.read()
    json_file.close()
    st.session_state['model'] = model_from_json(model_json)
    st.session_state['model'].load_weights("hand_gesture.weights.h5")

if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False

# Colors for displaying probabilities
colors = [(245, 117, 16) for _ in range(20)]

# Function to visualize probabilities
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), 
                      (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], 
                    (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Streamlit UI
st.title("Real-time Hand Gesture Recognition")

# Detection Variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

# Camera control button
if st.button("Toggle Camera", key="camera_toggle"):
    st.session_state['camera_active'] = not st.session_state['camera_active']

# Display current camera status
st.write(f"Camera is {'active' if st.session_state['camera_active'] else 'inactive'}.")

# Run the camera only if the session state says it's active
if st.session_state['camera_active']:
    st.text("Press the 'Toggle Camera' button to stop the webcam.")

    # Streamlit webcam input
    frame = st.camera_input("Camera", key="camera_input")

    if frame is not None:
        # Convert image from bytes to OpenCV format
        img = cv2.imdecode(np.frombuffer(frame.read(), np.uint8), cv2.IMREAD_COLOR)

        # Crop the frame for hand detection
        cropframe = img[40:400, 0:300]
        img = cv2.rectangle(img, (0, 40), (300, 400), (255, 255, 255), 2)

        # MediaPipe setup
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            
            # Make detections
            image, results = mediapipe_detection(cropframe, hands)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try:
                if len(sequence) == 30:
                    res = st.session_state['model'].predict(np.expand_dims(sequence, axis=0))[0]
                    predicted_action = actions[np.argmax(res)]
                    predictions.append(np.argmax(res))

                    # Display sentence and accuracy
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if predicted_action != sentence[-1]:
                                    sentence.append(predicted_action)
                                    accuracy.append(f"{res[np.argmax(res)] * 100:.2f}")
                            else:
                                sentence.append(predicted_action)
                                accuracy.append(f"{res[np.argmax(res)] * 100:.2f}")

                    if len(sentence) > 1:
                        sentence = sentence[-1:]
                        accuracy = accuracy[-1:]

                    # Display probability visualization on the frame
                    img = prob_viz(res, actions, img, colors, threshold)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

            # Display output text on the frame
            cv2.rectangle(img, (0, 0), (300, 40), (245, 117, 16), -1)
            cv2.putText(img, f"Output: {' '.join(sentence)} {''.join(accuracy)}", 
                        (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Update the frame in Streamlit
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

else:
    st.write("Press the 'Toggle Camera' button to start the webcam feed.")
