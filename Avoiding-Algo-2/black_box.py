import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# === Load and preprocess the image ===
image_path = 'target2.jpg'  # Use the same naming convention
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
input_image = image.astype(np.float32)
input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)
input_image = np.expand_dims(input_image, axis=0)

# === Simulate "Black Box" attack manually ===
# In this example, we simulate occlusion by placing a black rectangle over the eyes
adv_image = image.copy()
start_point = (60, 80)     # x, y coordinates
end_point = (170, 130)     # x, y coordinates for bottom-right of rectangle
color = (0, 0, 0)          # Black in RGB
thickness = -1             # Fill the rectangle

cv2.rectangle(adv_image, start_point, end_point, color, thickness)

# === Save the adversarial image ===
adv_image_bgr = cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("adversarial_output_blackbox.jpg", adv_image_bgr)
print("✅ Adversarial image (Black Box) saved as 'adversarial_output_blackbox.jpg'")

