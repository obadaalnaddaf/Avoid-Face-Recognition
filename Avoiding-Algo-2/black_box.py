import cv2
import numpy as np

# === Load and preprocess the image ===
image_path = 'target.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

# === Apply "Black Box" perturbation manually ===
# Example: draw a black rectangle across the eyes
adv_image = image.copy()
start_point = (50, 80)   # x, y
end_point = (170, 120)   # x, y
color = (0, 0, 0)        # Black in RGB
thickness = -1           # Fill the rectangle

cv2.rectangle(adv_image, start_point, end_point, color, thickness)

# === Save the adversarial image ===
adv_image_bgr = cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("adver.jpg", adv_image_bgr)

print("✅ Black Box applied. Saved as 'adver.jpg'")
