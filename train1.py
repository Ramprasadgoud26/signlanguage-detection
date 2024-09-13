import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from function import extract_keypoints

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions (gestures) that we try to recognize
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

# Number of sequences (videos) and sequence length (frames per video)
no_sequences = 300
sequence_length = 30

# Label mapping
label_map = {label: num for num, label in enumerate(actions)}

# Loading the training data
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                # Loading the .npy files containing keypoints, with allow_pickle=True
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
                
                # Ensure each frame has exactly 21 landmarks with 3 coordinates each (21 * 3 = 63 elements)
                if res.shape != (63,):
                    res = np.zeros(63)  # If the shape is not correct, fill with zeros
            except:
                # In case the .npy file is missing or corrupted, fill with zeros
                res = np.zeros(63)
            
            window.append(res)
        
        # Append the window (sequence) and label to our lists
        sequences.append(window)
        labels.append(label_map[action])

# Convert sequences and labels to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Print shapes to verify consistency
print(f"X shape: {X.shape}")  # Should be (180, 30, 63) for 180 sequences, 30 frames, and 63 keypoints per frame
print(f"y shape: {y.shape}")  # Should be (180, 6) for 180 sequences and 6 possible actions

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))  # Input shape must match (30, 63)
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Define a TensorBoard callback for logging
log_dir = os.path.join('Logs')
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tensorboard_callback], validation_data=(X_test, y_test))

# Save the model architecture to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("hand_gesture.weights.h5")

print("Model saved to disk successfully!")
