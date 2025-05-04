# DeepFace Face Recognition Test using FaceNet
# Then test PGD and DeepFool algorithm using the same model.

## Goal

To test **DeepFace** library using the **FaceNet** model on different face images (custom images).

---
### How to run the testing code:

## Step 1: Create virtual environment and Install Requirements

    1- python -m venv facenv

    2- Activate facenv environment:

    \path\to\environment\Scripts\activate

    3- pip install -r requirements.txt

    ---

## Step 2: Run model_test.py

    python model_test.py

## How To run Avoiding-Algo-1 (PGD):

## Step 1: Change the working directory to Avoiding-Algo-1:

## Step 2: Create virtual environment and Install Requirements:
    1- python -m venv pgd

    2- Activate pgd environment:

    \path\to\environment\Scripts\activate

    3- pip install -r requirements.txt


## Step 2: Run PGD.py

    python PGD.py

## Step 3: Testing:
    Once you run the PGD.py you will got a new image "adver.png" then copy it to "Face_Recognition_01/data_set/Jason_Statham/" and modify the code to test the new image.(Check line 33 in model_test.py "img1_path=resized_img1," ---> "img1_path=new_image,").
    Then Run the testing code again and check the results.

## How To run Avoiding-Algo-2 (DeepFool):

## Step 1: Change the working directory to Avoiding-Algo-2:

## Step 2: Create virtual environment and Install Requirements:
    1- python -m venv deepfool

    2- Activate deepfool environment:

    \path\to\environment\Scripts\activate

    3- pip install -r requirements.txt


## Step 2: Run DeepFool.py

    python DeepFool.py

## Step 3: Testing:
    Once you run the DeepFool.py you will got a new image "adver.png" then copy it to "Face_Recognition_01/data_set/Jason_Statham/" and modify the code to test the new image.(Check line 33 in model_test.py "img1_path=resized_img1," ---> "img1_path=new_image,").
    Then Run the testing code again and check the results.
