import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear,sigmoid


def display_digit(X):        
    fig, ax = plt.subplots(1,1, figsize=(0.5,0.5))   #create a plot for image
    X_reshaped = X.reshape((20,20)).T                #reshape it into 2D array and transpose it
    ax.imshow(X_reshaped, cmap='gray')               #Display the image in grayscale
    ax.set_axis_off()                                #To remove the axis
    plt.show()                                       #To diaplay in the screen
    

def len_errors(model,X,y):
    f = model.predict(X)
    yhat = np.argmax(f, axis=1)
    idxs = np.where(yhat != y[:,0])[0]
    return(len(idxs))


X=np.load("data/X.npy")
y=np.load("data/y.npy")

# print('The first element of X is',X[0])
# print('The first element of y is',y[0,0])
# print('The last element of y is',y[-1,0])
# print(X.shape)
# print(y.shape)

# display_digit(X[1])

m, n = X.shape
fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
for i,ax in enumerate(axes.flat):
    random_index = np.random.randint(m)                     # Select random indices
    X_random_reshaped = X[random_index].reshape((20,20)).T  # Select rows corresponding to the random indices and reshape the image
    ax.imshow(X_random_reshaped, cmap='gray')               # Display the image
    ax.set_title(y[random_index,0])                         # Display the label above the image
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)
plt.show()


tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),
        Dense(25,activation='relu',name='l1'),
        Dense(15,activation='relu',name='l2'),
        Dense(10,activation='linear',name='l3')
    ] 
)

# model.summary()

# [layer1, layer2, layer3] = model.layers
# W1,b1 = layer1.get_weights()
# W2,b2 = layer2.get_weights()
# W3,b3 = layer3.get_weights()
# print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
# print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
# print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
history = model.fit(
    X,y,
    epochs=50
)

#predicting a random number
# image_of_zero = X[0]
# display_digit(image_of_zero)
# prediction = model.predict(image_of_zero.reshape(1,400))  # prediction
# print(f" predicting a Two: \n{prediction}")
# print(f" Largest Prediction index: {np.argmax(prediction)}")
# prediction_p = tf.nn.softmax(prediction)                  #using softmax
# yhat = np.argmax(prediction_p)
# print(f"np.argmax(prediction_p): {yhat}")

predicting random numbers
m, n = X.shape
fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91])             #[left, bottom, right, top]
for i,ax in enumerate(axes.flat):
    random_index = np.random.randint(m)  
    X_random_reshaped = X[random_index].reshape((20,20)).T   
    ax.imshow(X_random_reshaped, cmap='gray')    
    prediction = model.predict(X[random_index].reshape(1,400)) # Predict using the Neural Network
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)    
    ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)    # Display the label above the image
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()

print(f'{len_errors(model,X,y)} errors out of {len(X)} items')