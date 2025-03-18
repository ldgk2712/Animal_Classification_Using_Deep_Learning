# Animal Classification Using Deep Learning

## 📌 Project Overview
This project focuses on classifying animals into 10 categories using deep learning techniques, specifically convolutional neural networks (CNN). The model is built using InceptionV3 and fine-tuned for optimal performance.

## 📁 Dataset
- **Source:** [Animals-10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data)
- **Description:** Contains ~28,000 images of 10 animal categories (dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant).
- **Split Ratio:**
  - Training: 80%
  - Validation: 10%
  - Testing: 10%

## 🏗️ Model Architecture
- **Base Model:** InceptionV3 (Pre-trained on ImageNet)
- **Modifications:**
  - Removed original classification head
  - Added Global Average Pooling layer
  - Added Fully Connected (Dense) layers
  - Final Softmax layer with 10 output classes

## 🔧 Data Preprocessing & Augmentation
- Rescaling pixel values to [0,1]
- Applying zoom transformations
- Horizontal flipping to increase dataset diversity

## 🏋️ Training Details
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Batch Size:** 64
- **Epochs:** 10
- **Callbacks Used:**
  - EarlyStopping (to prevent overfitting)
  - ModelCheckpoint (to save best model)
  - ReduceLROnPlateau (to adjust learning rate dynamically)

## 📊 Results
- **Test Accuracy:** 97%
- **Key Observations:**
  - Model successfully differentiates between species
  - Overfitting was mitigated through data augmentation and callbacks
  - Could be further improved with more diverse datasets and advanced architectures (e.g., Attention Mechanisms)

## 🚀 Future Improvements
- Expand dataset with more diverse images
- Explore Siamese Networks for fine-grained classification
- Implement real-time animal recognition system

## 📜 References
- [Animals-10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data)
- [Rethinking the Inception Architecture for Computer Vision (CVPR 2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
- [Popular Deep Learning Architectures](https://www.digitalocean.com/community/tutorials/popular-deep-learning-architectures-resnet-inceptionv3-squeezenet)

## 📌 How to Use
### Installation
```bash
pip install tensorflow keras numpy matplotlib
```
### Training the Model
```bash
python train.py
```
### Running Inference
```bash
python predict.py --image path_to_image.jpg
```

## 📌 Contributors
- **Lê Đặng Gia Khánh** - 22110081
- **Trần Anh Quốc** - 22110178
- **Nguyễn Bách Sơn** - 22110187

## 🔗 License
MIT License
