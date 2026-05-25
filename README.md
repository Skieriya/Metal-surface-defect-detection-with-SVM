# Metal Surface defect detection 

This repo has been updated and now uses MLflow for the model experiments 

# Why this?
Theres a popular iron mandi around my house and i grew up watching it since childhood so and have seen a lot of metal issues personally so i tried to apply my ML knowledge to solve one of the basic issue
which is obviosly metlal defects as large amounts of defects in metals cannot be shipped properly for large scale projects

# Approach 
Uses HOG and LBF features to train the MLflow pipeline of multiclass classification 
Features are scaled 
Uses a Label encoder 

# Dataset
https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
is used to train the models and contains a train and validation set

# Results

# Weighted precision
<img width="1200" height="600" alt="val_precision_weighted" src="https://github.com/user-attachments/assets/7cba984d-bfe4-4b67-a5fa-3e1327a0bbab" />

# Accuracy
<img width="1200" height="600" alt="validation_accuracy" src="https://github.com/user-attachments/assets/e28292d8-7db2-446f-b572-205545d626d5" />

# Weighted f1
<img width="1200" height="600" alt="val_f1_weighted" src="https://github.com/user-attachments/assets/9af8ca47-d31c-4fd0-9ee1-db05b160ec7f" />

# Streamlit app

<img width="1919" height="946" alt="image" src="https://github.com/user-attachments/assets/6c5d87b9-b073-4d4a-b24d-baea9fbec2ff" />



