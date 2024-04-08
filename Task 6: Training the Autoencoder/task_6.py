'''For training the Auto-encoder, we'll use the noisy training set examples as input and the original clean examples as labels to teach the model how to denoise. We'll set the number of epochs to 100 and utilize the early stopping callback.
To speed up training, we'll use a slightly larger batch size than usual. Additionally, we'll implement a lambda callback to log only the validation loss for each epoch. Setting the verbose parameter to False will hide unnecessary output and only display the validation loss per epoch.'''

autoencoder.fit(
    x_train_noisy,x_train,epochs=100,
    batch_size=512,validation_split = 0.2,verbose=False,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5),
        LambdaCallback(on_epoch_end=lambda e, l: print('{:,.3f}'.format(l['val_loss']), end='_'))
    
    ]
)



print('__')
print('Traing is complete')
