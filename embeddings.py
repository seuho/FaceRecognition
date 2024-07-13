import cv2

import pickle
import face_recognition_models
import face_recognition

name = input("Enter name: ")
ref_id = input("Enter ID: ")

try:
	f=open("ref_name.pkl","rb")

	ref_dictt=pickle.load(f)
	f.close()
except:
	ref_dictt={}
ref_dictt[ref_id]=name


f=open("ref_name.pkl","wb")
pickle.dump(ref_dictt,f)
f.close()

try:
	f=open("ref_embed.pkl","rb")

	embed_dictt=pickle.load(f)
	f.close()
except:
	embed_dictt={}





for i in range(10):
    key = cv2.waitKey(1)

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        break

    while True:
        check, frame = webcam.read()

        cv2.imshow("Capturing", frame)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            face_locations = face_recognition.face_locations(rgb_small_frame)

            if face_locations:
                face_encoding = face_recognition.face_encodings(frame)[0]
                if ref_id in embed_dictt:
                    embed_dictt[ref_id].append(face_encoding)
                else:
                    embed_dictt[ref_id] = [face_encoding]
                webcam.release()
                cv2.destroyAllWindows()
                break

        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            cv2.destroyAllWindows()
            break

# Save updated embeddings dictionary
with open("ref_embed.pkl", "wb") as f:
    pickle.dump(embed_dictt, f)
    f.close()