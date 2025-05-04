from deepface import DeepFace
import cv2
import os

# === Resize Function ===
def resize_image(input_path, output_path, width=250, height=250):
    """
    Resize an image to the given width and height and save it.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized)
    print(f"✅ Resized image saved to {output_path} ({width}x{height})")

# === Image Paths ===
original_img1 = "data_set/Jason_Statham/01.jpg"
original_img2 = "data_set/Jason_Statham/02.jpg"

# Paths for resized images
resized_img1 = "data_set/Jason_Statham/01_resized.jpg"
resized_img2 = "data_set/Jason_Statham/02_resized.jpg"
new_image = "data_set/Jason_Statham/adver.png"

# Resize both images
resize_image(original_img1, resized_img1)
resize_image(original_img2, resized_img2)

# === Run DeepFace Verification ===
result = DeepFace.verify(
    img1_path=resized_img1,
    img2_path=resized_img2,
    model_name="Facenet",
    enforce_detection=False
)

# === Output Results ===
print("🔍 Verified:", result["verified"])
print("📏 Distance:", result["distance"])
print("📦 Model used:", result["model"])
