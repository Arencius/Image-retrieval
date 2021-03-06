## About
A simple app for finding most similiar images in the CIFAR-10 dataset. Files in the repository:
- **model.py** - contains loading the dataset, building and training the CNN model
- **image_retrieval.py** - script for extracting the features of the images and finding the most similar photos
- **image_retrieval result.png** - screenshot of the results

## Model
The architecture of the Convolutional Neural Network used in this project to generate feature maps of the images consists of:
- **Residual blocks** - convolutional blocks with skip connections used in the original ResNet family architectures. Residual blocks are convolutional blokcs set in such a way that the output of a layer is taken and added to another layer deeper in the block
- **Squeeze and Excitation blocks** - according to the original SeNet paper it is an architectural unit designed to improve the representational power of a network by enabling it to perform dynamic channel-wise feature recalibration.
Implementation of SE block:

```
def squeeze_and_excitation_block(input_block, ch, ratio = 16):
    x = GlobalAveragePooling2D()(input_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    x = Reshape((1,1,ch))(x)
    
    return multiply([input_block, x])
```

Apart from model architecture, process of setting up and training the network included:
- data augmentation with custom implemented techniques, e.g. cropping or covering parts of the image,
- applying learning rate warm-up and cosine decay function to control the learning rate value throughout the training
- adding label smoothing to the loss function

Above techniques are derived from the paper: https://arxiv.org/abs/1812.01187

**After around 100 epochs of training, network's validation accuracy reached 93.86%.**

## Finding most similar images
Having trained the CNN classificator, the last layer with softmax activation is "cut off". After that the output of the model is the feature map generated by the GlobalAveragePooling2D layer.
After predicting the result (features) of the query image, a cosine distance between the rest of the images in the dataset is calculated. The most similar photos are the ones with the lowest distance between itself and the query photo.

## Results
![image](https://github.com/Arencius/Image-retrieval/blob/master/image_retrieval%20result.png)
