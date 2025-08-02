import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("models/best_model.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0) / 255.

pred = model.predict(img_tensor)
print("Predicted Class:", class_names[np.argmax(pred)])