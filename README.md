# Human-Recognition-in-Infrared-Images

This project was conducted in 2 stages.
The dataset can be found here- https://www.flir.com/oem/adas/adas-dataset-form/
## Stage 1-
The images from the dataset were segregated according to the attributes of the images provided in the dataset into human and non human images.
The features of the images were extracted using KAZE feature extractor in the OpenCV library for python.
These features were trained with the Weka tool for Windows into classifiers like random forest and logistic regression. Random Forest Classifier showed the maximum accuracy of 76%.

## Stage 2-
In this stage we created a custom dataset out of the images and the attributes provided. Each image was split into 12 parts and encoded as 1 or 0 whether or not a human is present in the image or not. 
This dataset was then passed through a 3-layer CNN.
This implementation increased our accuracy to 90%.
