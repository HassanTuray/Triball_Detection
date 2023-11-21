import os
from ultralytics import YOLO
import cv2

video_path = os.path.join('path to your video. Should be an mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter.fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('path to your trained yolo model')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5  # minimum score to detect tri-ball. A score of 0.75 means the model must be 75% sure of detection

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():  # unboxing results
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:  # checking if a tri-ball was detected
            # editing original frame to have bounding box around tribal localization
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)  # saving frame to a new video
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
