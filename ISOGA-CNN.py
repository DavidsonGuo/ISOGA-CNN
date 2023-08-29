#!/usr/bin/env python
# coding: utf-8

import os
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"  
import tensorflow as tf
#config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
#sess = tf.compat.v1.Session(config=config)
import pickle
import dill
import sys
from copy import deepcopy
from random import choice, randint, sample
from tensorflow import keras
import numpy as np
import pandas as pd

from tensorflow import (SGD, Adadelta, Adagrad, Adam, Adamax, Nadam,RMSprop)
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import to_categorical

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)      # suppress messages from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# topology
class Block:
	__slots__ = ('type', 'index', 'layerList1', 'layerList2')

	def __init__(self, type, index, layerList1, layerList2):
		self.type = type										# 0 -> initial layer; 1 -> mid layers; 2 -> final layer
		self.index = index										# block index among all the blocks
		self.layerList1 = layerList1							# Convolutional layers
		self.layerList2 = layerList2							# Pooling and Dropout layers

	def get_layers(self):
		return self.layerList1 + self.layerList2

	def get_size(self):
		return len(self.get_layers())

class Superpara:
    def __init__(self,batch_size,epochs,lr,opt):
        self.batch_size=batch_size
        self.epochs=epochs
        self.lr=lr
        self.opt=opt
    def build_layer(self,model):
        model.batch_size=self.batch_size
        model.epochs=self.epochs
        model.lr=self.lr
        if self.opt == 'Adadelta':
            model.opt = Adadelta(learning_rate=model.lr)
        elif self.opt == 'SGD':
            model.opt = SGD(learning_rate=model.lr)
        elif self.opt == 'RMSprop':
            model.opt = RMSprop(learning_rate=model.lr)
        elif self.opt == 'Adam':
            model.opt = Adam(learning_rate=model.lr)
        elif self.opt == 'Adagrad':
            model.opt = Adagrad(learning_rate=model.lr)
        elif self.opt == 'Adamax':
            model.opt = Adamax(learning_rate=model.lr)
        elif self.opt == 'Nadam':
            model.opt = Nadam(learning_rate=model.lr)
        
        print("model batch size has been set to ", model.batch_size)
        print("model epochs has been set to ", model.epochs)
        print("model lr has been set to ", model.lr)
        print("model optimizers has been set to ", self.opt)
    
    def mutate_parameters(self):
        print("Super Parameters Mutation")
        mutation = randint(0,5)
        if mutation == 0 and self.batch_size <= 200:
            print("-->changed batch_size from ", self.batch_size, " ", end="")
            self.batch_size = self.batch_size + 15
            print("to ", self.batch_size)
        elif mutation == 1 and self.batch_size >= 20:
            print("-->changed batch_size from ", self.batch_size, " ", end="")
            self.batch_size = self.batch_size - 15
            print("to ", self.batch_size)
        elif mutation == 2 and self.batch_size <= 200:
            print("-->changed batch_size from ", self.batch_size, " ", end="")
            self.batch_size = self.batch_size + 5
            print("to ", self.batch_size)
        elif mutation == 3 and self.batch_size >= 10:
            print("-->changed batch_size from ", self.batch_size, " ", end="")
            self.batch_size = self.batch_size - 5
            print("to ", self.batch_size)
        elif mutation == 4 and self.batch_size <= 200:
            print("-->changed batch_size from ", self.batch_size, " ", end="")
            self.batch_size = self.batch_size + 10
            print("to ", self.batch_size)
        elif mutation == 5 and self.batch_size >= 15:
            print("-->changed batch_size from ", self.batch_size, " ", end="")
            self.batch_size = self.batch_size - 10
            print("to ", self.batch_size)    
        
        
        mutation_1 = randint(0,5)
        if mutation_1 == 0 and self.epochs <= 200:
            print("-->changed epochs from ", self.epochs, " ", end="")
            self.epochs = self.epochs +20
            print("to ", self.epochs)
        elif mutation_1 == 1 and self.epochs >= 20:
            print("-->changed epochs from ", self.epochs, " ", end="")
            self.epochs = self.epochs -20
            print("to ", self.epochs)
        elif mutation_1 == 2 and self.epochs <= 200:
            print("-->changed epochs from ", self.epochs, " ", end="")
            self.epochs = self.epochs +5
            print("to ", self.epochs)
        elif mutation_1 == 3 and self.epochs >= 20:
            print("-->changed epochs from ", self.epochs, " ", end="")
            self.epochs = self.epochs -5
            print("to ", self.epochs)
        elif mutation_1 == 4 and self.epochs <= 200:
            print("-->changed epochs from ", self.epochs, " ", end="")
            self.epochs = self.epochs +10
            print("to ", self.epochs)
        elif mutation_1 == 5 and self.epochs >= 20:
            print("-->changed epochs from ", self.epochs, " ", end="")
            self.epochs = self.epochs -10
            print("to ", self.epochs)
        
        mutation_2 = randint(0,3)
        if mutation_2 == 0 and self.lr <= 0.01:
            print("-->changed lr from ", self.lr, " ", end="")
            self.lr = self.lr *2
            print("to ", self.lr)
        elif mutation_2 == 1 and self.lr <=  0.005:
            print("-->changed lr from ", self.lr, " ", end="")
            self.lr = self.lr *5
            print("to ", self.lr)
        elif mutation_2 == 2 and self.lr >=  0.00001:
            print("-->changed lr from ", self.lr, " ", end="")
            self.lr = self.lr /2
            print("to ", self.lr)
        elif mutation_2 == 3 and self.lr >=  0.00001:
            print("-->changed lr from ", self.lr, " ", end="")
            self.lr = self.lr /5
            print("to ", self.lr)
        
        mutation_3 = randint(0,6)
        if mutation_3 == 0 :
            print("-->changed opt from ", self.opt, " ", end="")
            self.opt = 'SGD'
            print("to ", self.opt)
        elif mutation_3 == 1 :
            print("-->changed opt from ", self.opt, " ", end="")
            self.opt = 'Adam'
            print("to ", self.opt)
        elif mutation_3 == 2 :
            print("-->changed opt from ", self.opt, " ", end="")
            self.opt = 'RMSprop'
            print("to ", self.opt)
        elif mutation_3 == 3 :
            print("-->changed opt from ", self.opt, " ", end="")
            self.opt = 'Adagrad'
            print("to ", self.opt)
        elif mutation_3 == 4 :
            print("-->changed opt from ", self.opt, " ", end="")
            self.opt = 'Adadelta'
            print("to ", self.opt)
        elif mutation_3 == 5 :
            print("-->changed opt from ", self.opt, " ", end="")
            self.opt = 'Adamax'
            print("to ", self.opt)
        elif mutation_3 == 6 :
            print("-->changed opt from ", self.opt, " ", end="")
            self.opt = 'Nadam'
            print("to ", self.opt)


class Convolutional:
	# __slots__ = ('name', 'filters', 'padding', 'filter_size', 'stride_size', 'input_shape')

	def __init__(self, filters, padding, filter_size, stride_size, input_shape):
		self.name = 'Conv1D'
		self.filters = filters
		self.padding = padding
		self.filter_size = filter_size
		self.stride_size = stride_size
		self.input_shape = input_shape

	def build_layer(self, model):
		model.add(tf.keras.layers.Conv1D(filters=self.filters,
									   kernel_size=self.filter_size,
									   strides=self.stride_size,
						               padding=self.padding,
									   activation='relu',
									   kernel_initializer='he_uniform',
									   input_shape=self.input_shape))

	def mutate_parameters(self):
		mutation = randint(0, 4)
		print("Mutating", self.name, "layer:")
		if mutation == 0 and self.filters >= 32:
			print("-->changed self.filters from ", self.filters, " ", end="")
			self.filters = int(self.filters / 2)
			print("to ", self.filters)
		elif mutation == 1 and self.filters >= 32:
			print("-->changed self.filters from ", self.filters, " ", end="")
			self.filters = int(self.filters / 2)
			print("to ", self.filters)
		elif mutation == 2 and self.filters <= 512:
			print("-->changed self.filters from ", self.filters, " ", end="")
			self.filters *= 2
			print("to ", self.filters)
		elif mutation == 3 and self.filters <= 512:
			print("-->changed self.filters from ", self.filters, " ", end="")
			self.filters *= 2
			print("to ", self.filters)
		elif mutation == 4:
			if self.padding == 'valid':
				print("-->changed self.padding from ", self.padding, " ", end="")
				self.padding = 'same'
				print("to ", self.padding)
			else:
				print("-->changed self.padding from ", self.padding, " ", end="")
				self.padding = 'valid'
				print("to ", self.padding)

class Pooling:
	__slots__ = ('name', 'pool_size', 'stride_size', 'padding')

	def __init__(self, pool_size, stride_size, padding):
		self.name = 'MaxPooling1D'
		self.pool_size = pool_size
		self.stride_size = stride_size
		self.padding = padding

	def build_layer(self, model):
		if self.name == 'MaxPooling1D':
			model.add(tf.keras.layers.MaxPooling1D(self.pool_size, self.stride_size, self.padding))
		elif self.name == 'AveragePooling1D':
			model.add(tf.keras.layers.AveragePooling1D(self.pool_size, self.stride_size, self.padding))

	def mutate_parameters(self):
		print("Mutating", self.name, "layer:")
		mutation = randint(0, 1)
		if mutation == 0:
			if self.padding == 'valid':
				print("-->changed self.padding from ", self.padding, " ", end="")
				self.padding = 'same'
				print("to ", self.padding)
			else:
				print("-->changed self.padding from ", self.padding, " ", end="")
				self.padding = 'valid'
				print("to ", self.padding)
		elif mutation == 1:
			if self.name == 'MaxPooling1D':
				print("-->changed self.name from ", self.name, " ", end="")
				self.name = 'AveragePooling1D'
				print("to ", self.name)
			else:
				print("-->changed self.name from ", self.name, " ", end="")
				self.name = 'MaxPooling1D'
				print("to ", self.name)

class FullyConnected:
    __slots__ = ('name', 'lstm_units', 'units', 'num_classes')

    def __init__(self, lstm_units, units, num_classes):
        self.name = "FullyConnected"
        self.units = units
        self.lstm_units = lstm_units
        self.num_classes = num_classes

    def build_layer(self, model):
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.units, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

    def mutate_parameters(self):
        print("Mutating", self.name, "layer:")
        mutation = randint(0, 2)
        if mutation == 0:
            print("-->changed self.units from ", self.units, " ", end="")
            self.units *= 2
            print("to ", self.units)
        elif mutation == 1:
            print("-->changed self.units from ", self.units, " ", end="")
            self.units *= 2
            print("to ", self.units)
        elif mutation == 2:
            print("-->changed self.units from ", self.units, " ", end="")
            self.units /= 2
            print("to ", self.units)
   

class Dropout:
	__slots__ = ('name', 'rate')

	def __init__(self, rate):
		self.name = "Dropout"
		self.rate = rate

	def build_layer(self, model):
		model.add(tf.keras.layers.Dropout(self.rate))

	def mutate_parameters(self):
		print("Mutating", self.name, "layer:")
		mutation = randint(0, 3)
		if mutation == 0 and self.rate <= 0.85:
			print("-->changed self.rate from ", self.rate, " ", end="")
			self.rate = self.rate + 0.10
			print("to ", self.rate)
		elif mutation == 1 and self.rate <= 0.90:
			print("-->changed self.rate from ", self.rate, " ", end="")
			self.rate = self.rate + 0.05
			print("to ", self.rate)
		elif mutation == 2 and self.rate >= 0.15:
			print("-->changed self.rate from ", self.rate, " ", end="")
			self.rate = self.rate - 0.10
			print("to ", self.rate)
		elif mutation == 3 and self.rate >= 0.10:
			print("-->changed self.rate from ", self.rate, " ", end="")
			self.rate = self.rate - 0.05
			print("to ", self.rate)

def load_dataset(num_classes):  					# import data
    df1 = pd.read_csv("train.csv")
    df1 = np.array(df1)
    X = np.expand_dims(df1[:, 1:891].astype(float), axis=2)#the data is augmented and converted to 32 bits
    Y = df1[:, 0]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.8,random_state=42)
    y_train = to_categorical(y_train, num_classes)  					# one-hot encoding
    y_test = to_categorical(y_test, num_classes)

    dataset = {
        #'batch_size': batch_size,
        'num_classes': num_classes,
        #'epochs': epochs,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

    return dataset


# save the model in binary form
def save_network(network):
    object_file = open(network.name + '.obj', 'wb')
    pickle.dump(network, object_file)


# read the model file in binary format
def load_network(name):
    object_file = open(name + '.obj', 'rb')
    return pickle.load(object_file)


# query the number of blocks
def order_indexes(self):
    i = 0
    for block in self.block_list:
        block.index = i
        i += 1

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

def plot_training(history):                                           # plot diagnostic learning curves
    plt.figure(figsize=[8, 6])											# loss curves
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation loss'])
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss curves', fontsize=16)
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_loss_plot.png')
    plt.close()

    plt.figure(figsize=[8, 6])											# accuracy curves
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy curves', fontsize=16)
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_acc_plot.png')
    plt.close()

def plot_statistics(stats):
    plt.figure(figsize=[8, 6])											# fitness curves
    plt.plot([s[0] for s in stats], 'r', linewidth=3.0)
    plt.plot([stats[0][0]] * len(stats), 'b', linewidth=3.0)
    plt.legend(['Best fitness', 'Initial fitness'])
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('Fitness value', fontsize=16)
    plt.title('Fitness curve', fontsize=16)
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_fitness_plot.png')
    plt.close()


    plt.figure(figsize=[8, 6])											# parameters curves
    plt.plot([s[1] for s in stats], 'r', linewidth=3.0)
    plt.plot([stats[0][1]] * len(stats), 'b', linewidth=3.0)
    plt.legend(['Best params number', 'Initial params number'])
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('Params number', fontsize=16)
    plt.title('Parameters curve', fontsize=16)
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_params_plot.png')
    plt.close()

def save_training_results(history, file_path):
    a = history.history['loss']
    b = history.history['val_loss']
    c = history.history['acc']
    d = history.history['val_acc']

    # save the training results as a dictionary
    results = {'loss': a, 'val_loss': b, 'acc': c, 'val_acc': d}

    # convert the dictionary to DataFrame format
    df = pd.DataFrame(results)

    # save the DataFrame as a csv file
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

class Network:
    __slots__ = ('name', 'block_list', 'fitness', 'model')

    def __init__(self, it):
        self.name = 'parent_' + str(it) if it == 0 else 'net_' + str(it)
        self.block_list = []
        self.fitness = None
        self.model = None
   
    def build_model(self):
        model = keras.Sequential()                                # create model
        for block in self.block_list:
            for layer in block.get_layers():                # build model
                try:
                    layer.build_layer(model)
                except:
                    print("\nINDIVIDUAL ABORTED, CREATING A NEW ONE\n")
                    return -1
        return model

    def train_and_evaluate(self, model, dataset):
        print("Training", self.name)
        model.compile(optimizer=model.opt, loss='categorical_crossentropy', metrics=['acc'])
        
        
        history = model.fit(dataset['x_train'],
                            dataset['y_train'],
                            batch_size=model.batch_size,
                            epochs=model.epochs,
                            validation_data=(dataset['x_test'], dataset['y_test']),
                            shuffle=True)

        
        self.model = model                                    # model
        self.fitness = history.history['val_acc'][-1]        # fitness

        print("SUMMARY OF", self.name)
        print(model.summary())
        print("FITNESS: ", self.fitness)
        model.save(self.name + '.h5') 
        model.save_weights(self.name+'weights'+'.h5') 
        print('================================================') 
        print('================================================') 
        # save model
        save_network(self) 
        with open(self.name+'.his', 'wb') as f:
            pickle.dump(history.history, f)
                                                                     # save topology, model and fitness

    def __getstate__(self):
        dic = {i: getattr(self, i) for i in self.__slots__ if i != 'model'}
        return dic
    
    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        from keras.models import load_model
        setattr(self, 'model', load_model(self.name + '.h5'))

    def asexual_reproduction(self, it, dataset):

        # if the individual already exists, just load it
        '''if os.path.isfile('net_' + str(it) + '.h5'):
            print("\n-------------------------------------")
            print("Loading individual net_" + str(it))
            print("--------------------------------------\n")
            individual = load_network('net_' + str(it))
            model = tf.keras.models.load_model(individual.name + '.h5')
            print("SUMMARY OF", individual.name)
            print(model.summary())
            print("FITNESS: ", individual.fitness)
            return individual'''

        # otherwise, create the individual by mutating the parent
        individual = Network(it)

        print("\n-------------------------------------")
        print("\nCreating individual", individual.name)
        print("--------------------------------------\n")

        individual.block_list = deepcopy(self.block_list)           # copy the layer list from parent

        print("----->Strong Mutation")
        individual.block_mutation(dataset)                          # mutate a block
        individual.layer_mutation(dataset)                          # mutate a layer
        individual.parameters_mutation()                            # mutate some parameters

        model = individual.build_model()

        if model == -1:
            return self.asexual_reproduction(it, dataset)

        individual.train_and_evaluate(model, dataset)

        return individual

    def block_mutation(self, dataset):
        print("Block Mutation")

        print([(block.index, block.type) for block in self.block_list])

        # block list containing all the blocks with type = 1
        bl = [block.index for block in self.block_list if block.type == 1]

        if len(bl) == 0:
            print("Creating a new block with two Convolutional layers and a Pooling layer")
            self.block_list[1].index = 2
            layerList1 = [
                Convolutional(filters=pow(2, randint(4, 7)),
                              filter_size=3,
                              stride_size=1,
                              padding='same',
                              input_shape=dataset['x_train'].shape[1:]),
                Convolutional(filters=pow(2, randint(4, 7)),
                              filter_size=3,
                              stride_size=1,
                              padding='same',
                              input_shape=dataset['x_train'].shape[1:])
            ]
            layerList2 = [
                Pooling(pool_size=2,
                        stride_size=2,
                        padding='same')
            ]
            b = Block(1, 1, layerList1, layerList2)     
            self.block_list.insert(1, b)
            return

        block_idx = randint(1, max(bl))         # pick a random block among all the blocks with type = 1
        block_type_idx = randint(0, 1)          # 1 -> Conv1D; 0 -> Pooling or Dropout
        mutation_type = randint(0, 1)           # 1 -> remove; 0 -> add

        # list of layers of the selected block
        layerList = self.block_list[block_idx].layerList1 if block_type_idx else self.block_list[block_idx].layerList2
        length = len(layerList)

        if mutation_type:                                       # remove，mytation type为1，删除层
            if length == 1:
                del self.block_list[block_idx]
            elif block_type_idx:
                pos = randint(0, length - 1)
                print("Removing a Conv1D layer at", pos)
                del layerList[pos]
            else:
                pos = randint(0, length - 1)
                print("Removing a Pooling/Dropout layer at", pos)
                del layerList[pos]
        else:                                                   # add
            if block_type_idx:
                print("Inserting a Convolutional layer")
                layer = Convolutional(filters=pow(2, randint(4, 7)),
                                      filter_size=3, 
                                      stride_size=1,
                                      padding='same',
                                      input_shape=dataset['x_train'].shape[1:])
                layerList.insert(randint(0, length - 1), layer)
            else:
                if randint(0, 1):                               # 1 -> Pooling; 0 -> Dropout
                    print("Inserting a Pooling layer")
                    layer = Pooling(pool_size=2,
                                    stride_size=2,
                                    padding='same')
                    layerList.insert(randint(0, length - 1), layer)
                else:
                    print("Inserting a Dropout layer")
                    rate = choice([0.15, 0.25, 0.35, 0.50])
                    layer = Dropout(rate=rate)
                    layerList.insert(randint(0, length - 1), layer)

    def layer_mutation(self, dataset):
        print("Layer Mutation")

        # pick a random block among all the blocks with type = 1
        bl = [block.index for block in self.block_list if block.type == 1]

        if len(bl) == 0:
            return

        block_idx = randint(1, max(bl))
        block_type_idx = randint(0, 1)      # 1 -> Conv2D; 0 -> Pooling or Dropout

        # list of layers of the selected block
        layerList = self.block_list[block_idx].layerList1 if block_type_idx else self.block_list[block_idx].layerList2

        if len(layerList) == 0:
            if block_type_idx:
                layer = Convolutional(filters=pow(2, randint(4, 7)),
                                      filter_size=3,
                                      stride_size=1,
                                      padding='same',
                                      input_shape=dataset['x_train'].shape[1:])
                self.block_list[block_idx].layerList1.append(layer)
                return
            else:
                layer = Pooling(pool_size=2,
                                stride_size=2,
                                padding='same')
                self.block_list[block_idx].layerList2.append(layer)

        idx = randint(0, len(layerList) - 1)
        layer = layerList[idx]

        if layer.name == 'Conv1D':
            print("Splitting Conv1D layer at index", idx)
            layer.filters = int(layer.filters * 0.5)
            layerList.insert(idx, deepcopy(layer))
        elif layer.name == 'MaxPooling1D' or layer.name == 'AveragePooling1D':
            print("Changing Pooling layer at index", idx, "with Conv1D layer")
            del layerList[idx]
            conv_layer = Convolutional(filters=pow(2, randint(4, 7)),
                                       filter_size=3,
                                       stride_size=1,
                                       padding=layer.padding,
                                       input_shape=dataset['x_train'].shape[1:])
            layerList.insert(idx, conv_layer)

    def parameters_mutation(self):
        print("Parameters Mutation")
        for block in self.block_list:
            for layer in block.get_layers():
                if randint(0, 1):
                    layer.mutate_parameters()


def compute_parent(dataset):
    '''if os.path.isfile('parent_0.h5'):
        daddy = load_network('parent_0')
        model = tf.keras.models.load_model('parent_0.h5')
        print("Loading parent_0")
        print("SUMMARY OF", daddy.name)
        print(model.summary())
        print("FITNESS:", daddy.fitness)
        return daddy'''
   
    daddy = Network(0)
    
    layerList1 = [
        Convolutional(filters=16, filter_size=3, stride_size=1, padding='same',
                      input_shape=dataset['x_train'].shape[1:]),
        Convolutional(filters=32, filter_size=3, stride_size=1, padding='valid',
                      input_shape=dataset['x_train'].shape[1:])
    ]
    layerList2 = [
        Pooling(pool_size=2, stride_size=2, padding='same')
    ]
    daddy.block_list.append(Block(0, 0, layerList1, layerList2))

    layerList1 = [
        Convolutional(filters=64, filter_size=3, stride_size=1, padding='same',
                      input_shape=dataset['x_train'].shape[1:]),
        Convolutional(filters=256, filter_size=3, stride_size=1, padding='valid',
                      input_shape=dataset['x_train'].shape[1:])
    ]
    layerList2 = [
        Pooling(pool_size=4, stride_size=2, padding='same'),
        
    ]
    daddy.block_list.append(Block(1, 1, layerList1, layerList2))
    
    
    layerList1 = [
        FullyConnected(lstm_units=1024, units=128, num_classes=dataset['num_classes'])
    ]
    layerList2 = [Superpara(batch_size=25,epochs=100,lr=0.0005,opt='RMSprop')]
    daddy.block_list.append(Block(2, 2, layerList1, layerList2))
    
    model = daddy.build_model()
    print('====================>')
    print(model)
    daddy.train_and_evaluate(model, dataset)
    return daddy

def initialize_population(population_size, dataset):
    print("----->Initializing Population")
    daddy = compute_parent(dataset)                                 # load parent from input
    population = [daddy]
    for it in range(1, population_size):
        population.append(daddy.asexual_reproduction(it, dataset))

    # sort population on ascending order based on fitness
    return sorted(population, key=lambda cnn: -cnn.fitness)

def selection(k, population, num_population):
    if k == 0:                                              # elitism selection
        print("----->Elitism selection")
        return population[0], population[1]
    elif k == 1:                                            # tournament selection
        print("----->Tournament selection")
        i = randint(0, num_population - 1)
        j = i
        while j < num_population - 1:
            j += 1
            if randint(1, 100) <= 50:
                return population[i], population[j]
        return population[i], population[0]
    else:                                                   # proportionate selection
        print("----->Proportionate selection")
        cum_sum = 0
        for i in range(num_population):
            cum_sum += population[i].fitness
        perc_range = []
        for i in range(num_population):
            count = int(100 * population[i].fitness / cum_sum)
            for j in range(count):
                perc_range.append(i)
        i, j = sample(range(1, len(perc_range)), 2)
        while i == j:
            i, j = sample(range(1, len(perc_range)), 2)
        return population[perc_range[i]], population[perc_range[j]]


def crossover(parent1, parent2, it):
    print("----->Crossover")
    child = Network(it)

    first, second = None, None
    if randint(0, 1):
        first = parent1
        second = parent2
    else:
        first = parent2
        second = parent1

    child.block_list = deepcopy(first.block_list[:randint(1, len(first.block_list) - 1)]) \
                       + deepcopy(second.block_list[randint(1, len(second.block_list) - 1):])

    order_indexes(child)                            # order the indexes of the blocks

    return child


def genetic_algorithm(num_population, num_generation, num_offspring, dataset):
    print("Genetic Algorithm")

    population = initialize_population(num_population, dataset)

    print("\n-------------------------------------")
    print("Initial Population:")
    for cnn in population:
        print(cnn.name, ': ', cnn.fitness)
    print("--------------------------------------\n")

    # for printing statistics about fitness and number of parameters of the best individual
    stats = [(population[0].fitness, population[0].model.count_params())]

    for gen in range(1, num_generation + 1):

        k = randint(0, 2)

        print("\n------------------------------------")
        print("Generation", gen)
        print("-------------------------------------")

        for c in range(num_offspring):

            print("\nCreating Child", c)

            parent1, parent2 = selection(k, population, num_population)                 # selection
            print("Selected", parent1.name, "and", parent2.name, "for reproduction")

            child = crossover(parent1, parent2, c + num_population)                     # crossover
            print("Child has been created")

            print("----->Soft Mutation")
            child.layer_mutation(dataset)                                               # mutation
            child.parameters_mutation()
            print("Child has been mutated")

            model = child.build_model()                                                 # evaluation
            

            while model == -1:
                child = crossover(parent1, parent2, c + num_population)
                child.block_mutation(dataset)
                child.layer_mutation(dataset)
                child.parameters_mutation()
                model = child.build_model()

            child.train_and_evaluate(model, dataset)
            with open(str(gen)+'gen'+str(c)+'child_his'+'.his', 'wb') as f:
                pickle.dump(model.history.history, f)
            model.save(str(gen)+'gen'+str(c)+'child_model'+'.h5')
            model.save_weights(str(gen)+'gen'+str(c)+'child_weights'+'.h5')

            if child.fitness > population[-1].fitness:                                  # evolve population
                print("----->Evolution: Child", child.name, '第几代',str(gen),'第几个孩子' ,str(c),"with fitness", child.fitness, "replaces parent ", end="")
                print(population[-1].name, "with fitness", population[-1].fitness)
                name = population[-1].name
                population[-1] = deepcopy(child)
                population[-1].name = name
                population = sorted(population, key=lambda net: -net.fitness)
            else:
                print("----->Evolution: Child", child.name, "with fitness", child.fitness, "is discarded")

        stats.append((population[0].fitness, population[0].model.count_params()))

    print("\n\n-------------------------------------")
    print("Final Population")
    print("-------------------------------------\n")
    for cnn in population:
        print(cnn.name, ': ', cnn.fitness)

    print("\n-------------------------------------")
    print("Stats")
    for i in range(len(stats)):
        print('+++++++++++++++===============+++++++++++++++')
        print("Best individual at generation", i + 1, "has fitness", stats[i][0], "and parameters", stats[i][1])
    print("-------------------------------------\n")

    # plot the fitness and the number of parameters of the best individual at each iteration
    plot_statistics(stats)

    return population[0]




def main():
    #batch_size = 32                         # the number of training examples in one forward/backward pass
    num_classes = 6                        # number of cifar-10 dataset classes
    #epochs = 1                              # number of forward and backward passes of all the training examples
    
    with open('output.txt', 'w') as f:         #all output results are saved in txt==========
        sys.stdout = f

        dataset = load_dataset(num_classes)
        
        num_population = 12
        num_generation = 8
        num_offspring = 3
        # plot the best model obtained
        optCNN = genetic_algorithm(num_population, num_generation, num_offspring, dataset)
        
        # plot the training and validation loss and accuracy
        model = optCNN.build_model()
        model.compile(optimizer=model.opt, loss='categorical_crossentropy', metrics=['acc'])
        print('======================')
        print('======================')
        print('======================')
        print("Current lr and opt are",model.lr, model.opt)
        history = model.fit(dataset['x_train'],
                            dataset['y_train'],
                            batch_size=model.batch_size,
                            epochs=model.epochs,
                            validation_data=(dataset['x_test'], dataset['y_test']),
                            shuffle=True)
        optCNN.model = model                                        # model
        optCNN.fitness = history.history['val_acc'][-1]            # fitness
        with open('optCNN.his', 'wb') as f:
            pickle.dump(history.history, f)

        model.save('optCNN.h5')

        print("\n\n-------------------------------------")
        print("The initial CNN has been evolved successfully in the individual", optCNN.name)
        print("-------------------------------------\n")
        daddy = load_network('parent_0')
        model = tf.keras.models.load_model('parent_0.h5')
        print("\n\n-------------------------------------")
        print("Summary of initial CNN")
        print(model.summary())
        print("Fitness of initial CNN:", daddy.fitness)

        print("\n\n-------------------------------------")
        print("Summary of evolved individual")
        print("================================")
        print("================================")
        print("================================")
        print(optCNN.model.summary())
        print("Current lr and opt are",optCNN.model.lr,optCNN.model.opt)
        print("Current batch_size and epochs is",optCNN.model.batch_size,optCNN.model.epochs)
        print("Fitness of the evolved individual:", optCNN.fitness)
        print("-------------------------------------\n")

        plot_training(history)
        save_training_results(history, 'training_results.csv') 

if __name__ == '__main__':
    main()
    





