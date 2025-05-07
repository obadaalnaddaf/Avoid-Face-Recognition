**Summary of References**

**1. FaceNet: A Unified Embedding for Face Recognition and Clustering**
•	Authors: Florian Schroff, Dmitry Kalenichenko, James Philbin
•	Link: https://arxiv.org/abs/1503.03832
This paper introduces FaceNet, a deep learning model that maps face images into a compact Euclidean space using a triplet loss function, enabling direct comparison based on similarity. It achieved groundbreaking performance in clustering and face recognition tasks, notably 99.63% accuracy on LFW, using only 128-dimensional embeddings.

**2. DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks**
•	Authors: Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard
•	Link: https://arxiv.org/abs/1511.04599
DeepFool is a method that calculates minimal perturbations needed to change a classifier’s decision. The algorithm works iteratively and efficiently to estimate a sample's proximity to the decision boundary, thereby evaluating a model’s robustness against adversarial attacks.

**3. Persistent Classification: Understanding Adversarial Attacks by Studying Decision Boundary Dynamics**
•	Authors: Brandon Bell, Anastasios Angelopoulos, Nicholas Carlini, Michael Jordan
•	Link: https://arxiv.org/abs/2404.08069
This paper presents a novel method called persistent classification that studies the geometric behavior of decision boundaries under adversarial perturbations. Using topological tools like persistent homology, it reveals deeper understanding of data stability and adversarial dynamics.

**4. You Only Attack Once: Single-Step DeepFool Algorithm**
•	Authors: Sajjad Khodabandelou, Mohammad Ali Zolfaghari
•	Link: https://www.mdpi.com/2076-3417/15/1/302
This work improves upon DeepFool by proposing a single-step version, significantly reducing computational time while maintaining adversarial effectiveness. It emphasizes efficiency in generating adversarial examples without compromising success rates.

**5. Explaining and Harnessing Adversarial Examples**
•	Authors: Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy
•	Link: https://arxiv.org/abs/1412.6572
A foundational paper explaining why adversarial examples exist, proposing the Fast Gradient Sign Method (FGSM) and demonstrating the impact of adversarial training to enhance model robustness.

**6. Practical Black-Box Attacks against Deep Learning Systems Using Adversarial Examples**
•	Authors: Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, et al.
•	Link: https://arxiv.org/abs/1602.02697
This research shows how adversarial attacks can be executed in black-box settings using substitute models. It proves that adversarial examples can transfer across different architectures, enabling attacks without direct model access.

**7. Adversarial Examples in the Physical World**
•	Authors: Alexey Kurakin, Ian Goodfellow, Samy Bengio
•	Link: https://arxiv.org/abs/1607.02533
The authors examine adversarial robustness in the physical world, demonstrating that printed adversarial images remain effective even when photographed. This highlights real-world vulnerabilities of vision systems.

**8. Detection and Prevention of Evasion Attacks on Machine Learning Models**
•	Authors: M. Nasrullah, M. K. Rafique, M. Arif, et al.
•	Link: https://www.sciencedirect.com/science/article/abs/pii/S0957417424029117
A comprehensive survey of evasion attacks and defense mechanisms, discussing detection, input transformation, robust training, and hybrid methods to defend against adversarial manipulation.

**9. OpenCV Documentation: OpenCV Modules**
•	Authors: OpenCV Community
•	Link: https://docs.opencv.org/
OpenCV is a powerful open-source library for real-time computer vision. The documentation outlines its capabilities for image processing, including face detection, feature extraction, and manipulation — essential for pre-processing in face recognition pipelines.

**10. Towards Evaluating the Robustness of Neural Networks**
Authors: Nicholas Carlini, David Wagner
Link: https://arxiv.org/abs/1608.04644
This paper introduces the Carlini & Wagner (C&W) attack, one of the most effective optimization-based methods for generating adversarial examples. It emphasizes the need for better evaluation metrics and stronger baselines for adversarial robustness, setting a new standard in adversarial testing.
