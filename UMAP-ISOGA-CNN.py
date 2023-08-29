import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import umap.umap_ as umap 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential 
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense 
from keras.utils import to_categorical 
import keras

df1 = pd.read_csv('spectral.csv', header=None)
df1 = np.array(df1)

X = np.expand_dims(df1[:, 1:891].astype(float), axis=2) # add a channel dimension
y = df1[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train) # one-hot encoding
y_test = to_categorical(y_test)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

model = keras.models.load_model('optCNN.h5')
print(model.summary())
layer_names = ['Conv1d 1', 'Conv1d 2', 'Average pooling1D 1', 'Conv1d 3', 'Conv1d 4', 'Conv1d 5',  'Dropout 1',
               'Conv1d 6', 'Conv1d 7', 'Conv1d 8', 'Dropout 2','Flatten', 'Fully connected 1', 'Fully connected 2']
y_labels = ['Control', 'Day 1', 'Day 3', 'Day 5', 'Day 7', 'Day 9'] # add all label names here

for i in range(len(layer_names)):
    model_layer = Sequential()
    for layer in model.layers[:i+1]: # include up to i-th layer
        model_layer.add(layer)
    embeddings = model_layer.predict(X)
    print('embeddings==========')
    print(embeddings.shape)
    if len(embeddings.shape) == 3:
        embeddings = np.reshape(embeddings, (embeddings.shape[0], embeddings.shape[1]*embeddings.shape[2]))
        print('reshape embeddings==========')
        print(embeddings.shape)
    umap_embeddings = umap.UMAP(n_components=3).fit_transform(embeddings)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in range(len(y_labels)):
        ax.scatter(umap_embeddings[y==label, 0], umap_embeddings[y==label, 1], umap_embeddings[y==label, 2], label=y_labels[label])
    ax.set_title(layer_names[i])
    plt.subplots_adjust(bottom=0.1,left=0.1,top=0.9,right=0.9)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(layer_names[i]+ 'UMAP.tif')
    plt.show()
    if i == len(layer_names) - 1:
        break # exit loop after last layer