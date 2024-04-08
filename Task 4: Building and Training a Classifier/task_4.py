'''In this task, we're building a classifier to recognize handwritten digit images. We'll use a simple neural network with two hidden layers, each having 256 nodes. The output layer will have 10 nodes, one for each digit class (0 to 9), and we'll apply a softmax function to get probability scores for each class.
One thing to note is that we'll use sparse categorical cross-entropy loss instead of regular categorical cross-entropy loss. This is because our labels are not one-hot encoded; instead, they're just numbers from 0 to 9, each representing a digit class. So, with sparse categorical cross-entropy, we can handle these numerical labels directly without needing to one-hot encode them.'''


classifier = Sequential([
    Dense(256,activation='relu',input_shape=(784,)),
    Dense(256,activation='relu'),
    Dense(10,activation='softmax')
])
classifier.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', #lebels are not encorded,these are just numeric values
    metrics=['accuracy']
)

classifier.fit(x_train,y_train,batch_size=512,epochs =3)


loss,acc=classifier.evaluate(x_test,y_test)
print(acc)

loss,acc=classifier.evaluate(x_test_noisy,y_test)
print(acc)
