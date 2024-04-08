'''To reduce noise in our data, we're creating an Auto-encoder model. This model takes a noisy example as input and the corresponding original, clean example as the label. If we design one or more hidden layers in this neural network to have significantly fewer nodes compared to the input and output layers, the training process will naturally push the network to learn a function similar to principal component analysis (PCA), effectively reducing dimensionality.

It's important to note that the output layer of the Auto-encoder has the sigmoid activation function. This choice is suitable because our input examples are black and white images. With sigmoid activation, higher linear values in the last layer will tend towards the maximum normalized pixel value of 1, while lower linear values will converge towards the minimum normalized pixel value of 0. Since most of the pixel values in black and white images are either 0 (black) or 1 (white), using sigmoid activation helps focus the model's attention on these extreme values, which works well for our task.'''

input_image = Input(shape=(784,))
encoded=Dense(64, activation='relu')(input_image)

decoded=Dense(784,activation='sigmoid')(encoded)

autoencoder=Model(input_image,decoded)
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

