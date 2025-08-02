import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("models/best_model.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (150, 150))
    img = img_to_array(img) / 255.
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label = class_names[np.argmax(pred)]

    cv2.putText(frame, f"{label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Waste Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()