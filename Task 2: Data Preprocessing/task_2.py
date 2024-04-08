''' 

For this project, we'll be using the well-known MNIST dataset, which contains 60,000 training examples and 10,000 test examples of handwritten digits. Each image is represented as a 28x28 pixel grayscale image, meaning it has 28 rows and 28 columns. The labels for these images are simply the digits from 0 to 9, corresponding to the classes.

We'll create two neural network models for this project:

1. Classification Model: This model will be trained to classify the handwritten digits into their respective classes (0 to 9).
2. Auto-encoder: This model will be used to denoise input data. It will take noisy images as input and produce clean, denoised images as output.

Eventually, we'll combine these two models into a single composite model. This composite model will take noisy images as input, denoise them using the Auto-encoder, and then classify the denoised images using the classification model.

To prepare the data for input into our models, we'll perform some preprocessing steps, including:
- Reshaping the images to the appropriate dimensions.
- Normalizing the pixel values to a range between 0 and 1.
- One-hot encoding the labels for the classification model.
- Preparing noisy and clean versions of the input data for the Auto-encoder model.

By completing these preprocessing steps, we'll ensure that our data is in the correct format and ready to be fed into our neural network models. Let's proceed with the data pre-processing.'''

(x_train,y_train),(x_test , y_test) = mnist.load_data() #numpy arrays
#for normalization
x_train = x_train.astype('float')/255.
x_test = x_test.astype('float')/255.
x_train = np.reshape(x_train,(60000,784)) #784=28*28
x_test = np.reshape(x_test,(10000,784))
