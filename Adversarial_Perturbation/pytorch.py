import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# === Load Pretrained ResNet18 ===
model = models.resnet18(pretrained=True)
model.eval()

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load and preprocess image
img = Image.open("data_set/Jason_Statham/01.jpg")
input_img = transform(img).unsqueeze(0)  # Add batch dimension

# === Define Target Class ===
# For example, target class = 859 ("toaster") from ImageNet
target_class = torch.tensor([859])

# === Enable Gradient Calculation ===
input_img.requires_grad = True

# === Forward Pass ===
output = model(input_img)

# === Calculate Loss (Targeted Attack) ===
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, target_class)

# === Backward Pass ===
model.zero_grad()
loss.backward()

# === FGSM Attack ===
epsilon = 0.03  # small perturbation magnitude
data_grad = input_img.grad.data
sign_data_grad = data_grad.sign()
perturbed_image = input_img - epsilon * sign_data_grad
perturbed_image = torch.clamp(perturbed_image, 0, 1)

# === Predict Again ===
output_adv = model(perturbed_image)
_, pred_adv = torch.max(output_adv, 1)

# === Show Results ===
def imshow(tensor, title=""):
    image = tensor.squeeze().detach().numpy()
    image = np.transpose(image, (1, 2, 0))  # CHW to HWC
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

imshow(input_img, title="Original Image")
imshow(perturbed_image, title=f"Adversarial Image (Predicted class: {pred_adv.item()})")


from torchvision.transforms.functional import to_pil_image

# === Save Adversarial Image ===
adv_image = to_pil_image(perturbed_image.squeeze())  # Remove batch dimension
adv_image_path = "data_set/Jason_Statham/01_adv.jpg"
adv_image.save(adv_image_path)
print(f"⚠️ Adversarial image saved to {adv_image_path}")