# AI application for plants classification

This project involves developing an image classifier to recognize different species of flowers, 
using a dataset of 102 flower categories. The classifier can be integrated into applications like
a smartphone app that identifies flowers through a camera. The model is trained and exported for 
easy integration into real-world software solutions.

## Tech stack
- Python 3.0
- PyTorch
- PIL
- JSON

## Data

For training, there are transformations like random 
scaling, cropping, and flipping, and resized all images to 224x224 pixels. For validation and testing,
there are resized and cropped the images without transformations. Additionally, the images have been 
normalized using ImageNet's mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225] 
to match the network's input expectations. The data can  [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

## Running the application
```shell script
   python train.py --data_dir "./flowers/"
   python train.py data_dir --arch "vgg13"
   python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20

   python predict.py /path/to/image checkpoint
   python predict.py --image_file "flowers/test/1/image_06743.jpg"

```
## Features

## **Image Preprocessing**
   - **Random Scaling**: Images are resized with random scaling to introduce variation during training.
   - **Random Cropping**: Random cropping is applied to the images to capture different parts of the flowers and prevent overfitting.
   - **Flipping**: Horizontal flipping is used as a transformation technique to add diversity to the training set.
   - **Resizing**: All images are resized to a fixed size of 224x224 pixels for consistency.
   - **Normalization**: The images are normalized using ImageNet’s mean and standard deviation to match the network's expected input:
     - Mean: `[0.485, 0.456, 0.406]`
     - Standard Deviation: `[0.229, 0.224, 0.225]`

## **Dataset Transformation**
   - **Training Set**: Images are subject to random transformations (scaling, cropping, and flipping) to ensure robustness during training.
   - **Validation Set**: Images are resized and cropped but not randomly transformed to provide a stable evaluation of the model's performance.
   - **Test Set**: Images are resized and cropped, with no transformations applied, ensuring a consistent evaluation environment.

## **Model Architecture**
   - **Pre-Trained Network**: A pre-trained network (e.g., on ImageNet) is used for feature extraction, leveraging previously learned features for flower classification.
   - **Custom Classifier**: A new, untrained feed-forward network is added on top of the pre-trained model. This classifier uses ReLU activations and dropout layers to prevent overfitting.
   - **Backpropagation**: The classifier layers are trained using backpropagation to learn the task-specific features.

## **Model Training**
   - **Loss Tracking**: The model's loss is tracked throughout the training process to monitor progress and avoid overfitting.
   - **Accuracy Tracking**: Accuracy on the validation set is monitored to assess the model’s performance and tune hyperparameters effectively.
   - **Hyperparameter Tuning**: The best hyperparameters (such as learning rate, batch size, and number of epochs) are selected by evaluating loss and accuracy on the validation set.

## **Saving the Model**
   - **Checkpoint Saving**: After training, the model is saved as a checkpoint, which includes the model's learned weights and parameters, as well as the mapping of flower categories to class indices. This enables future predictions and model reuse.

## **Loading the Model**
   - **Checkpoint Loading**: The saved checkpoint is used to load the trained model to avoid retraining it.
   - **Image Loading**: The images are loaded using the Python Imaging Library (PIL) and are resized to 256 pixels, followed by a center crop of 224x224 pixels.
   - **Image Normalization**: The loaded image is normalized to a [0, 1] range and adjusted using ImageNet’s mean and standard deviation for consistency with the training data.

## **Class Prediction**
   - **Top 5 Class Prediction**: The model predicts the top 5 most probable flower classes based on the input image.
   - **Prediction Visualization**: The predicted classes and their associated probabilities are visualized, allowing users to see the model’s confidence in each class.