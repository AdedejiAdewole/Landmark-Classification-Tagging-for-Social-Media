# Landmark-Classification-Tagging-for-Social-Media

The main objectice of this project is to build models to automatically predict the location of the image based on any landmarks depicted in an image.

# Project
Build a landmark classifier

## 1. Overview
This project was a part of the project assessment in the 'AWS x Udacity's Machine Learning Engineer Nanodegree Scholarship Program'.

## 2. Getting Started
### 2.1 Project Files
1. cnn_from_scratch.ipynb: This jupyter notebook contains codes to check if the environment is setup correctly, download the data if it's not downloaded already, and also check that your GPU is available and ready to go. It is in this notebook that the source files required to this project are tested. These tests are to help in finding obvious probelms that are in our source files and a way to fix them. Lastly, in this notebook, a CNN is created from scratch to classify landmarks.

2. transfer_learning.ipynb: This jupyter notebook uses tranfer learning to create a CNN for classifying landmark images as opposed to building a CNN from scratch. Differnt pretrained models are investigated and one is selected to use for classification.

3. app.ipynb: In this notebook, the best model is used to create a simple app for others to be able to use your model to find the most likely landmarks depicted in an image.

4. Source files: These are python scripts that contains functions that are used to create the models reqqured for this classification.
   1. data.py: This preprocesses the data used for classifications.
   2. model.py: This defines the CNN architecture.
   3. optimization.py: This defines the loss and optimizer of our model.
   4. predictor.py: This contains codes to wrap the model for inference.
   5. train.py: This contains codes to train, optimize while training and perform validations on our models.
   6. transfer.py: This contains codes for using transfer learning to create a CNN for classifying landmark images.

5. cnn_from_scratch.html: Web-page displaying 'cnn_from_scratch.ipynb'.

6. transfer.html: Web-page displaying 'transfer.ipynb'.

7. app.html: Web-page displaying 'app.ipynb'.

### 2.2 Dependencies
You can develop your project locally on your computer. This setup requires a bit of familiarity with creating a working deep learning environment. While things should work out of the box, in case of problems you might have to do operations on your system (like installing new NVIDIA drivers for your GPU). 

1. Download the starter kit from the Resources section of the left sidebar of the Udacity classroom here. (Scroll down if you don't see it.)
2. Open a terminal and navigate to the directory where you installed the starter kit.
3. Download and install Miniconda
4. Create a new conda environment with Python 3.7.6:
```
conda create --name udacity python=3.7.6 pytorch=1.11.0 torchvision torchaudio cudatoolkit -c pytorch
```
5. Activate the environment:
```
conda activate udacity
```
NOTE: You will need to activate your environment again every time you open a new terminal.

6. Install the required packages for the project:
```
pip install -r requirements.txt
```
7. Test that the GPU is working (execute this only if you have a NVIDIA GPU on your machine, which Nvidia drivers properly installed)
```
python -c "import torch;print(torch.cuda.is_available())
```
This should return True. If it returns False your GPU cannot be recognized by pytorch. Test with nvidia-smi that your GPU is working. If it is not, check your NVIDIA drivers.

8. Install and open jupyter lab:
```
pip install jupyterlab 
jupyter lab
```

### 2.3 Building the landmark classification models
#### 2.3.1 CNN from scratch

![E9C5FC78-42EB-431D-A5D4-5A20FAC89AC4_1_201_a](https://github.com/AdedejiAdewole/Landmark-Classification-Tagging-for-Social-Media/assets/50617984/34bbe40f-3b82-4e96-8ea0-98c434746902)


#### 2.3.2 CNN from scratch after hyperparametization 

![D5697938-28C5-4FFA-8951-8BEBC18CDD04_1_201_a](https://github.com/AdedejiAdewole/Landmark-Classification-Tagging-for-Social-Media/assets/50617984/7307d3dd-94f8-47c0-ba1e-d20fdb7b9f4c)

![90359397-D825-4F22-81A7-C0594988CA6C_1_201_a](https://github.com/AdedejiAdewole/Landmark-Classification-Tagging-for-Social-Media/assets/50617984/89598751-d7ab-4249-bcb5-7e099b721cb2)


#### 2.3.2 Transfer Learning

![21ACC74A-F456-4EA6-BA86-0A923C144721_1_201_a](https://github.com/AdedejiAdewole/Landmark-Classification-Tagging-for-Social-Media/assets/50617984/2e298965-1ad8-4849-87a6-11ba7c545f0d)




