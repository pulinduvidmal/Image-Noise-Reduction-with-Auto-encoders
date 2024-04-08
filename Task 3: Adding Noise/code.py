'''
We add noise to our training and test examples on purpose. But why? Well, in real life, our data often has some unwanted noise, like static on a TV screen. But here's the catch: we usually don't have perfect, noise-free versions of our data to compare against.
So, to teach our Auto-encoder model how to deal with this noisy data, we first create our own noise and add it to clean images. This way, we have pairs of images: one clean and one noisy. We use these pairs to train the Auto-encoder to focus on the important parts of the images, even when there's noise.
Then, when we give the Auto-encoder real-world data with noise, it already knows how to handle it. It knows where to look and which details are crucial, even if the data isn't perfect. So, by training with artificially added noise, we're helping the Auto-encoder get better at handling noisy data in the real world.
'''


x_train_noisy = x_train +np.random.rand(60000,784)*1
x_test_noisy = x_test +np.random.rand(10000,784)*1
x_train_noisy=np.clip(x_train_noisy,0.,1.)
x_test_noisy = np.clip(x_test_noisy,0.,1.)


def plot(x,p,labels=False):   #p=prediction
    plt.figure(figsize=(20,2))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.imshow(x[i].reshape(28,28), cmap= 'binary')
        plt.xticks([])
        plt.yticks([])
        if labels:
            plt.xlabel(np.argmax(p[i]))
    plt.show()
    
plot(x_train,None)
