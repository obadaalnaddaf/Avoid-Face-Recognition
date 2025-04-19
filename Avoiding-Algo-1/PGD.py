import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load pre-trained model (you can replace this with your own)
model = tf.keras.applications.MobileNetV2(weights='imagenet')
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Load and preprocess image
image_path = 'target.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
input_image = preprocess(image.astype(np.float32))
input_image = np.expand_dims(input_image, axis=0)

# PGD parameters
eps = 2          # Max perturbation
alpha = 0.02        # Step size
iterations = 200

# Convert to tensor
adv_image = tf.convert_to_tensor(input_image)

# Record original label
true_label = tf.argmax(model.predict(adv_image)[0])

# PGD attack
for i in range(iterations):
    with tf.GradientTape() as tape:
        tape.watch(adv_image)
        prediction = model(adv_image)
        true_label = int(tf.argmax(model.predict(adv_image)[0]).numpy())
        loss = tf.keras.losses.sparse_categorical_crossentropy(np.array([true_label]), prediction)


    # Get gradient
    gradient = tape.gradient(loss, adv_image)
    signed_grad = tf.sign(gradient)

    # Apply step
    adv_image = adv_image + alpha * signed_grad
    adv_image = tf.clip_by_value(adv_image, input_image - eps, input_image + eps)
    adv_image = tf.clip_by_value(adv_image, -1.0, 1.0)  # keep in valid range for MobileNet


#show_images(input_image[0], adv_image.numpy()[0])
# Save the adversarial image
adv_image_np = adv_image.numpy()[0]
# Convert from [-1, 1] to [0, 255]
adv_image_np = ((adv_image_np + 1) * 127.5).astype(np.uint8)

# Convert from RGB to BGR since OpenCV expects BGR
adv_image_bgr = cv2.cvtColor(adv_image_np, cv2.COLOR_RGB2BGR)

# Save to disk
cv2.imwrite("adversarial_output.png", adv_image_bgr)
print("✅ Adversarial image saved as 'adversarial_output.png'")

