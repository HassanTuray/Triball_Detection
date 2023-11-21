import os
from ultralytics import YOLO
import cv2

url = ("your url here")

model_path = os.path.join('path to your trained model here')

model = YOLO(model_path)  # load custom model

threshold = 0.75  # minimum score to detect tri-ball. A score of 0.75 means the model must be 75% sure of detection

cap = cv2.VideoCapture(url)  # getting feed from phone camera.

while (cap.isOpened()):

    ret, frame = cap.read()

    results = model(frame)[0]

    for result in results.boxes.data.tolist():  # unboxing results
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:  # checking if a tri-ball was detected
            # editing original frame to have bounding box around tribal localization
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Name of window", cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
    if ret == False:
        break

    key = cv2.waitKey(1)

    if key == ord('q'):  # if the q key is pressed then the window is closed and program stops
        break

    if key == ord('p'):  # if the p key is pressed then camera feed and detection is paused
        cv2.waitKey(-1)


cap.release()
cv2.destroyAllWindows()
