import face_recognition
import cv2
import numpy as np
import pickle

# Load reference names dictionary
try:
    with open("ref_name.pkl", "rb") as f:
        ref_dict = pickle.load(f)
except FileNotFoundError:
    ref_dict = {}

# Load reference embeddings dictionary
try:
    with open("ref_embed.pkl", "rb") as f:
        embed_dict = pickle.load(f)
except FileNotFoundError:
    embed_dict = {}

# Initialize known face encodings and names
known_face_encodings = []
known_face_names = []

for ref_id, embed_list in embed_dict.items():
    for embed in embed_list:
        known_face_encodings.append(embed)
        known_face_names.append(ref_id)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Adjust these values to increase/decrease the size of the rectangle
RECTANGLE_PADDING = 20

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Process every other frame
    if process_this_frame:
        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Increase the size of the rectangle
        top = max(0, top - RECTANGLE_PADDING)
        right = min(frame.shape[1], right + RECTANGLE_PADDING)
        bottom = min(frame.shape[0], bottom + RECTANGLE_PADDING)
        left = max(0, left - RECTANGLE_PADDING)

        # Choose color based on whether the face is recognized or not
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        label = ref_dict.get(name, name)
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
