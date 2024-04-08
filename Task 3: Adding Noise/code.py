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
