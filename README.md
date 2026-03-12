
# 🫁 Lung Cancer Detection Using CNN

##  Information
Made by: Angel Jain  
Project Title: Lung Cancer Detection Using Convolutional Neural Network (CNN)

#  Project Description

Lung cancer is one of the leading causes of death worldwide. Early detection plays a very important role in improving survival rates.

This project focuses on building an Artificial Intelligence system that can analyze CT scan images of lungs and detect whether the lung is normal or cancer affected.

The system uses a **Convolutional Neural Network (CNN)** to learn patterns from CT scan images and classify them into two categories:

- Cancer
- Normal

A user-friendly interface has also been developed using **Streamlit**, where users can upload CT scan images and instantly receive predictions.

## Methodology

1. CT scan images were collected and divided into two classes: **Cancer** and **Normal**.  
2. Images were resized to **128×128 pixels** and normalized before training.  
3. A **Convolutional Neural Network (CNN)** model was built using TensorFlow/Keras.  
4. The model learns important features from images through convolution and pooling layers.  
5. After training, the model predicts whether a lung CT scan is **Normal or Cancerous**.  
6. A Streamlit web interface** was developed to upload CT images and display predictions with confidence scores.

# Results

The trained CNN model successfully classifies CT scan images into:

- Normal Lung
- Cancer Detected

The system provides:

- Prediction result
- Confidence percentage
- Graph showing probability distribution

The model achieved high accuracy during testing and can effectively identify patterns in CT scan images.

<img width="1600" height="492" alt="image" src="https://github.com/user-attachments/assets/3ce6f239-c566-4347-8995-1d09bc51fa4e" />
<img width="1600" height="842" alt="image" src="https://github.com/user-attachments/assets/291d211c-554a-42ce-86ba-a44b394386bc" />



## Note

The trained model file is not uploaded due to GitHub file size limitations.
To generate the model locally, run:-  python train_model.py
This will create the trained model file:-  model/lung_cancer_model.h5
After that run the application:-  streamlit run app.py

