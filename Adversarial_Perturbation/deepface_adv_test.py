import cv2
import numpy as np
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
from deepface import DeepFace
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# === Load and preprocess image ===
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === Image paths ===
img_path = "data_set/Jason_Statham/01_resized.jpg"
target_img_path = "data_set/Jason_Statham/02_resized.jpg"
output_path = "data_set/Jason_Statham/01_adv_art.jpg"

# === Preprocess images ===
x = preprocess(img_path)
target = preprocess(target_img_path)

# === Load DeepFace Facenet model ===
facenet_model = DeepFace.build_model("Facenet")

# === Wrap Facenet model with ART classifier ===
classifier = KerasClassifier(model=facenet_model.model, clip_values=(0.0, 1.0))


# === Create FGSM attack ===
attack = FastGradientMethod(estimator=classifier, eps=0.02)
x_adv = attack.generate(x=x)

# === Save adversarial image ===
adv_img_np = (x_adv[0] * 255).astype(np.uint8)
adv_img_np = cv2.cvtColor(adv_img_np, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, adv_img_np)
print(f"⚔️ ART-generated FGSM adversarial image saved to: {output_path}")
