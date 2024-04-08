'''For obtaining de-noised images, particularly for our test data, we simply need to pass the noisy data through the trained Auto-encoder. We can achieve this by utilizing the predict method on our Auto-encoder model.
Once we have the de-noised images, we can pass them through our classifier. Since the Auto-encoder has learned to remove noise from the images, our classifier should perform notably better on these de-noised images compared to the original noisy ones.'''

predictions = autoencoder.predict(x_test_noisy)
plot(x_test_noisy,None)
plot(predictions,None)

loss , acc =classifier.evaluate(predictions,y_test)
print(acc)
