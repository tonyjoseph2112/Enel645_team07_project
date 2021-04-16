#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
get_ipython().run_line_magic('matplotlib', 'qt')
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Activation,Input
from tensorflow.keras.optimizers import SGD


# In[2]:


fpath = r"C:\Users\tonyj\Downloads\archive\PlantVillage"
random_seed = 111

categories = os.listdir(fpath)
print("List of categories = ",categories,"\n\nNo. of categories = ", len(categories))


# In[3]:


def load_images_and_labels(categories):
    img_lst=[]
    labels=[]
    for index, category in enumerate(categories):
        for image_name in os.listdir(fpath+"/"+category)[300:]:
            file_ext = image_name.split(".")[-1]
            if (file_ext.lower() == "jpg") or (file_ext.lower() == "jpeg"):
                #print(f"\nCategory = {category}, Image name = {image_name}")
                img = cv2.imread(fpath+"/"+category+"/"+image_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img_array = Image.fromarray(img, 'RGB')

                #resize image 100 X 100
                resized_img = img_array.resize((100, 100))

                img_lst.append(np.array(resized_img))

                labels.append(index)
    return img_lst, labels

images, labels = load_images_and_labels(categories)
print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))
print(type(images),type(labels))


# In[4]:


images = np.array(images)
labels = np.array(labels)

print("Images shape = ",images.shape,"\nLabels shape = ",labels.shape)
print(type(images),type(labels))


# In[5]:


# Here some random images are displayed
def display_rand_images(images, labels):
    plt.figure(1 , figsize = (19 , 10))
    n = 0 
    for i in range(9):
        n += 1 
        r = np.random.randint(0 , images.shape[0] , 1)
        
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)
        plt.imshow(images[r[0]])
        
        plt.title('Plant label : {}'.format(labels[r[0]]))
        plt.xticks([])
        plt.yticks([])
        
    plt.show()
    
display_rand_images(images, labels)


# In[7]:


(X_dev, Y_dev)=(images,labels)
indexes = np.arange(X_dev.shape[0], dtype = int)
np.random.shuffle(indexes)
X_dev = X_dev[indexes]
Y_dev = Y_dev[indexes]

nsplit1 = int(0.75*X_dev.shape[0]) # Train/validation split
nsplit2 = int(0.85*X_dev.shape[0]) # Test/validation split

# Train and validation split
X_train = X_dev[:nsplit1]
Y_train = Y_dev[:nsplit1]
X_val = X_dev[nsplit1:nsplit2]
Y_val = Y_dev[nsplit1:nsplit2]
# Test set
X_test = X_dev[nsplit2:]
Y_test = Y_dev[nsplit2:]

print("\nTrain set")
print("Images: ",X_train.shape)#prints the total no of images in the training set
print("Labels shape: ",Y_train.shape)
print("\nValidation set")
print("Images: ",X_val.shape)#prints the total no of images in the validation set
print("Labels shape: ",Y_val.shape)
print("\n Test set")
print("Images: ",X_test.shape)#prints the total no of images in the test set
print("Labels shape: ",Y_test.shape)


# In[8]:


# Data scaling
norm_type=0
if norm_type==0:
   X_train=X_train/255
   X_val=X_val/255
   X_test=X_test/255
elif norm_type==1:
     train_mean=X_train.mean()
     train_std =X_train.std()
     X_train=(X_train-train_mean)/(X_train-train_std)
     X_val=(X_val-train_mean)/(X_train-train_std)
     X_test=(X_test-train_mean)/(X_train-train_std)
else: 
    pass
    


# In[9]:


# One hot encoding
Y_train_oh=to_categorical(Y_train)
Y_val_oh=to_categorical(Y_val)
Y_test_oh=to_categorical(Y_test)
print(Y_train[:5])
print(Y_train_oh[:5])
k=np.unique(Y_dev).size
print(k)


# In[10]:


# Data Augmentation
batch_size = 32
# gen_params is the input parameter given to the ImageDataGenerator function
gen_params = {"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'constant',               "cval": 0}
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params)# Creates batches of  image data with real time data augmentation with the parameters specified above
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params)

train_gen.fit(X_train,seed = 1)
val_gen.fit(X_val, seed = 1)

train_flow = train_gen.flow(X_train,Y_train_oh,batch_size = batch_size)# the flow function is called where the input parameters are the training labels and batch size
val_flow = val_gen.flow(X_val,Y_val_oh,batch_size = batch_size)# the above function is called with validation labels


# In[11]:


# Displaying some samples from the development set
plt.figure(figsize = (32,24))
Xbatch,Ybatch = train_flow.__getitem__(0)
print(Xbatch.mean(),Xbatch.std())
print(Xbatch.min(),Xbatch.max())
# Images are plotted after data augmentation
for ii in range(batch_size):
    plt.subplot(7,5,ii+1)
    plt.imshow((Xbatch[ii]- Xbatch[ii].min())/(Xbatch.max() - Xbatch[ii].min()), cmap = "gray")
    plt.title("Label: %s" %categories[int(Ybatch[ii].argmax())])
plt.show()


# In[12]:


def my_model(ishape = (100,100,3),k = 15, lr = 1e-4):#defines a function named my_model which creates a CNN model as specified
    model_input = tf.keras.layers.Input(shape = ishape)
    l1 = tf.keras.layers.Conv2D(48, (3,3), padding='same', activation='relu')(model_input)
    l1_drop = tf.keras.layers.Dropout(0.25)(l1)
    l2 = tf.keras.layers.MaxPool2D((2,2))(l1_drop)
    l3 = tf.keras.layers.Conv2D(96, (3,3), padding='same', activation='relu')(l2)
    l4 = tf.keras.layers.Conv2D(96, (3,3), padding='same', activation='relu')(l3)
    l4_drop = tf.keras.layers.Dropout(0.25)(l4)
    flat = tf.keras.layers.Flatten()(l4_drop)
    out = tf.keras.layers.Dense(k,activation = 'softmax')(flat)
    model = tf.keras.models.Model(inputs = model_input, outputs = out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics = ["accuracy"])
    return model


# In[13]:


# defining callbacks
# First, We create an early-stop that monitors the validation loss and stops the training after 20 epochs if there is no improvement to prevent overfitting.
model_name="leafdisease_classification.h5"
early_stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=20)
monitor=tf.keras.callbacks.ModelCheckpoint(model_name,monitor="val_loss",                                          verbose=0,save_best_only=True,                                          save_weights_only=True,mode='min')
# We then define a learning rate scheduler that reduces the learning rate by half after every 10 epochs
def scheduler(epoch,lr):
    if epoch%10==0:
        lr=lr/2
    return lr
lr_schedule=tf.keras.callbacks.LearningRateScheduler(scheduler)


# In[14]:


model=my_model()
model.summary()#gives the summary of the model
# We train our model for 5 epochs with a batch size of 32
model.fit(X_train,Y_train_oh,batch_size=32,epochs=5,verbose=1,callbacks=[early_stop,monitor,lr_schedule],
          validation_data=(X_val,Y_val_oh),shuffle=True)


# In[15]:


model.load_weights(model_name)
metrics = model.evaluate(X_test,Y_test_oh)


# In[16]:


pred = model.predict(X_test)

pred.shape


# In[17]:


# code that displays images which were wrongly predicted
plt.figure(1 , figsize = (19 , 10))
n = 0 

for i in range(9):
    n += 1 
    r = np.random.randint( 0, X_test.shape[0], 1)
    
    plt.subplot(3, 3, n)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    plt.imshow(X_test[r[0]])
    plt.title('Actual = {}, Predicted = {}'.format(Y_test[r[0]] , Y_test[r[0]]*pred[r[0]][Y_test[r[0]]]) )
    plt.xticks([]) , plt.yticks([])

plt.show()


# In[ ]:




