import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === Load model ===
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# === Load and preprocess image ===
image_path = 'target1.jpg'  
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_image = transform(image).unsqueeze(0).detach().requires_grad_()

# === Enhanced DeepFool attack ===
def deepfool(image, model, num_classes=5, overshoot=18, max_iter=100, scaling_factor=23):
    image = image.clone().detach().requires_grad_()
    f_image = model(image).detach().numpy().flatten()
    I = f_image.argsort()[::-1][:num_classes]
    label = I[0]

    pert_image = image.clone()
    r_tot = torch.zeros_like(image)
    loop_i = 0
    x = pert_image.clone().detach().requires_grad_()
    fs = model(x)
    k_i = label

    while k_i == label and loop_i < max_iter:
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.clone()

        min_pert = float('inf')
        w = torch.zeros_like(x)

        for k in range(1, num_classes):
            x.grad.data.zero_()
            fs[0, I[k]].backward(retain_graph=True)
            grad_cur = x.grad.data.clone()

            w_k = grad_cur - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data
            pert_k = torch.abs(f_k) / torch.norm(w_k.flatten())

            if pert_k < min_pert:
                min_pert = pert_k
                w = w_k

        r_i = (min_pert + 1e-4) * w / torch.norm(w.flatten())
        r_tot = r_tot + r_i
        pert_image = image + (1 + overshoot) * scaling_factor * r_tot
        x = pert_image.clone().detach().requires_grad_()
        fs = model(x)
        k_i = fs.detach().numpy().flatten().argsort()[::-1][0]
        loop_i += 1

    return pert_image

# === Generate and save adversarial image ===
adv_image = deepfool(input_image, model)
adv_image_np = adv_image.squeeze().detach().numpy()
adv_image_np = np.transpose(adv_image_np, (1, 2, 0))
adv_image_np = np.clip(adv_image_np, 0, 1)
plt.imsave("adversarial_output_deepfool.jpg", adv_image_np)
print("✅ Stronger adversarial image saved as 'adversarial_output_deepfool.jpg'")

# === Visual check (optional)
diff = (adv_image - input_image).squeeze().detach().numpy()
diff = np.transpose(diff, (1, 2, 0))

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(np.transpose(input_image.squeeze().detach().numpy(), (1, 2, 0)))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Adversarial")
plt.imshow(adv_image_np)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow((diff * 10 + 0.5).clip(0, 1))
plt.axis("off")

plt.tight_layout()
plt.show()
