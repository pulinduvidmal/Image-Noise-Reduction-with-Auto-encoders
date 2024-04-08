'''To create a composite model for our prediction pipeline, we'll combine the trained Auto-encoder and the classifier into a single model. This composite model will take a noisy image as input, pass it through the Auto-encoder to reduce noise, and then feed the de-noised image into the classifier to get the class prediction.

Here's how we can create this composite model:

1. Define the input layer for the noisy image.
2. Pass the noisy image through the trained Auto-encoder to get the de-noised image.
3. Define the output layer for the classifier.
4. Combine the Auto-encoder and classifier into a single model.

This composite model will enable us to simply feed a noisy image, and it will automatically perform noise reduction using the Auto-encoder before making predictions with the classifier.'''

input_image=Input(shape=(784,))
x=autoencoder(input_image)
y=classifier(x)


denoise_and_classify = Model(input_image , y)

predictions = denoise_and_classify.predict(x_test_noisy)

plot(x_test_noisy,predictions,True)

plot(x_test,to_categorical(y_test),True)
