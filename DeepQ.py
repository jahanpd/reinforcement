import numpy as np
from sumtree import SumTree
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Flatten, Dense
from tensorflow.keras.losses import mean_squared_error as mse
from customlayer import equationNine

class DuelDeepQ:
	def __init__(self, timesteps, x, actions):
	# define values like self.value = blah
		self.timestep = timesteps
		self.actions = actions # number of actions possible

		self.xshape = x.shape
		self.dimensions = self.xshape[0] # define dimensionality of data
		self.train = x[:,0:int(np.round(0.8*self.xshape[1]))] # training set
		self.test = x[:,int(np.round(0.8*self.xshape[1])):] # test set

		# set default hyperparameters
		self.layers = 4
		self.nodes = 20
		self.gamma = 0.9
		self.prob = 0.1 # probability of random action during training


	def build(self):
		self.ANN = self.buildANN(self.layers,self.nodes) # build dense ANN architecture w 4 layers and 20 nodes in each layer
		print(self.ANN.summary())

	def buildANN(self, layers, nodes):
		# build initial input network
		inputs = Input(shape=(self.timestep,self.dimensions), name="data_in")
		y_true = Input(shape=(self.actions,), name="y_true")
		f = Flatten()(inputs)
		f = Dense(nodes, activation=tf.nn.tanh)(f)
		for layer in range(layers):
			f = Dense(nodes, activation=tf.nn.tanh)(f)

		# build state value network
		V = Dense(10, activation=tf.nn.tanh)(f)
		V = Dense(10, activation=tf.nn.tanh)(V)
		state = Dense(1, activation=tf.nn.leaky_relu)(V) # outputs a scalar value for state advantage

		# build action advantage network
		A = Dense(10, activation=tf.nn.tanh)(f)
		A = Dense(10, activation=tf.nn.tanh)(A)
		actions = Dense(self.actions, activation=tf.nn.leaky_relu)(A) # outputs a vector with advantage value of each action

		# final layer aka equationNine from http://proceedings.mlr.press/v48/wangf16.pdf
		y_pred = equationNine(self.actions)([actions,state])

		return Model(inputs = inputs, outputs=y_pred, name="deep neural net")

	# =============
	# MODEL TRAINING
	# ============

	# data preparation functions
	def _train_chunk(self):
		samples = np.empty([self.dimensions,0])
		for step in range(self.xshape[1] - self.timestep):
			samples = np.append(samples, self.train[:,step:step+timestep])
		return samples.reshape(self.dimensions, self.xshape[1] - self.timestep, self.timestep)

	# define training hyperparameters differently to default above
	def hyperparameters(self,
						newlayers = None, 
						newnodes = None, 
						newgamma = None,
						newprob = None):
		if self.layers == None:
			self.layers = newlayers
		if self.nodes == None:
			self.nodes = newnodes
		if self.gamma == None:
			self.gamma = newgamma
		if self.prob == None:
			self.prob = newprob

	# training function with inputs as hyperparameters
	def training(self, epochs):
		replay = SumTree(epochs * (self.xshape[1] - self.timestep))
		training_samples = self._train_chunk()
		
		self.cash = [1000]
        self.position = [0]
        self.value =[1000]
		for step in range(self.xshape[1] - self.timestep - 1):
			state = training_samples[step]
			next_state = training_samples[step+1]
			
			delta = next_state[0][0][-1] - state[0][0][-1]
			
			actionq = self.ANN.predict(np.expand_dims(state.T,0))[0]
			action = np.argmax(actionq)
			maxq = np.max(self.ANN.predict(np.expand_dims(next_state.T,0)))
			target = actionq
			
			if action == 0:
				reward = _a_is_0(state[0][0][-1])
				target[action] = (self.gamma * maxq) + reward
			else:
				reward = _a_is_1(state[0][0][-1])
				target[action] = (self.gamma * maxq) + reward


	def _a_is_0(self, s):
        if self.cash[-1] == 0:
        	self.cash.append(self.position[-1]*s)
            self.position.append(0)
            self.value.append(self.cash[-1])
            return self.value[-1] - self.value[-2]
        else:
            self.cash.append(self.cash[-1])
            self.position.append(0)
            self.value.append(self.cash[-1])
            return self.value[-1] - self.value[-2]
	def _a_is_1(self, s):
    	if position[-1] == 0:
        	self.position.append(self.cash[-1]/s)
        	self.cash.append(0)
        	self.value.append(self.position[-1]*s)
        	return self.value[-1] - self.value[-2]
	    else:
	        self.position.append(self.position[-1])
	        self.value.append(self.position[-1]*s)
	        return self.value[-1] - self.value[-2]






