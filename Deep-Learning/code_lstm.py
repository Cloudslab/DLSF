import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import preprocessing
import os.path
import pickle

loss_parameters = np.load('loss_parameters.npy')
cnn_parameters = np.load('cnn_parameters.npy')
lstm_parameters = np.load('lstm_parameters.npy')
#3,7,8
loss_values = []
for l in loss_parameters:
	loss_values += [(l[3] + l[7] + l[8])/10000000]
loss_values = np.array(loss_values)
# print(loss_values)

y_values = np.random.randint(10, size=len(lstm_parameters))
# print(y_values)

cnn_parameters_flat= cnn_parameters.reshape((-1,cnn_parameters.shape[1]*cnn_parameters.shape[2]))
lstm_parameters_flat = lstm_parameters.reshape((-1,lstm_parameters.shape[1]*lstm_parameters.shape[2]))

lstm_parameters_preprocessed = preprocessing.normalize(lstm_parameters_flat)
cnn_parameters_preprocessed = preprocessing.normalize(cnn_parameters_flat)

cnn_parameters = cnn_parameters_preprocessed.reshape((-1,cnn_parameters.shape[1],cnn_parameters.shape[2]))
lstm_parameters = lstm_parameters_preprocessed.reshape((-1,lstm_parameters.shape[1],lstm_parameters.shape[2]))

features_train_cnn = cnn_parameters[:int(0.8*cnn_parameters.shape[0])]
features_test_cnn = cnn_parameters[int(0.8*cnn_parameters.shape[0]):]

features_train_lstm = lstm_parameters[:int(0.8*lstm_parameters.shape[0])]
features_test_lstm = lstm_parameters[int(0.8*lstm_parameters.shape[0]):]

targets_train = y_values[:int(0.8*y_values.shape[0])]
targets_test = y_values[int(0.8*y_values.shape[0]):]

loss_train = loss_values[:int(0.8*y_values.shape[0])]
loss_test = loss_values[int(0.8*y_values.shape[0]):]

featuresTrainCNN = torch.from_numpy(features_train_cnn).type(torch.FloatTensor)
featuresTestCNN = torch.from_numpy(features_test_cnn).type(torch.FloatTensor)
featuresTrainLSTM = torch.from_numpy(features_train_lstm).type(torch.FloatTensor)
featuresTestLSTM = torch.from_numpy(features_test_lstm).type(torch.FloatTensor)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
lossTrain = torch.from_numpy(loss_train).type(torch.FloatTensor)
lossTest = torch.from_numpy(loss_test).type(torch.FloatTensor)

batch_size = 5
n_iters = 10000
num_epochs = n_iters / (len(features_train_cnn) / batch_size)
num_epochs = int(num_epochs)

train = torch.utils.data.TensorDataset(featuresTrainCNN,featuresTrainLSTM,targetsTrain,lossTrain)
test = torch.utils.data.TensorDataset(featuresTestCNN,featuresTestLSTM,targetsTest,lossTest)

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


input_dim = 15
hidden_dim = 26
num_layers = 2
output_dim = 100

learning_rate = 0.001

seq_dim = 1  
loss_list = []
iteration_list = []
accuracy_list = []
PATH = './models/'

no_of_hosts = 100
no_of_vms = 100

class DeepRL(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim, batch_size):
		super(RNNModel, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.output_dim = output_dim
		self.hidden = []

		for i in range(no_of_hosts):
			self.hidden += [self.init_hidden()]

		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
		# self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
		self.conv1 = nn.Conv2d(1, 10, kernel_size=2)
		self.conv2 = nn.Conv2d(10, 1, kernel_size=3)
		self.conv3 = nn.Conv2d(10, 10, kernel_size=2)

		self.fc1 = nn.Linear(88, 1000)
		self.fc2 = nn.Linear(1000, 5000)
		self.fc3 = nn.Linear(5000, 10000)

		self.bn1 = nn.BatchNorm1d(26)

	def init_hidden(self):
		return (torch.zeros(self.num_layers,self.batch_size, self.hidden_dim),
				torch.zeros(self.num_layers,self.batch_size, self.hidden_dim))

	def forward(self, cnn_data, lstm_data):
		lstm_out_host = []
		for i in range(lstm_data.shape[1]):
			# print(lstm_data[:,i,:])
			out, hidden = self.lstm(lstm_data[:,i,:].view(-1,1,lstm_data.shape[2]))
			self.hidden[i] = hidden
			lstm_out_host += [out]
			# lstm_out_host += [self.bn1(out.view(-1,out.shape[2])).view(-1,1,out.shape[1])]

		# lstm_out = np.array(lstm_out)
		# print(lstm_out_host[0].shape)
		lstm_out = lstm_out_host[0]
		for i in range(1,lstm_data.shape[1]):
			lstm_out = torch.cat((lstm_out,lstm_out_host[i]),1)
		# print(lstm_out)
		# print(self.hidden[0].shape)
		x1 = lstm_out.reshape((lstm_out.shape[0],1,lstm_out.shape[1],lstm_out.shape[2]))
		x1 = F.relu(F.max_pool2d(self.conv1(x1),2))
		x1 = self.conv2(x1)

		cnn_data = cnn_data.reshape((-1,1,cnn_data.shape[1],cnn_data.shape[2]))
		x2 = F.relu(F.max_pool2d(self.conv1(cnn_data),2))
		x2 = self.conv2(x2)

		x3 = torch.cat((x1,x2),2)

		x3 = self.conv1(x3)
		x3 = F.max_pool2d((x3),2)
		x3 = self.conv2(x3)

		x3 = x3.reshape(-1,x3.shape[2]*x3.shape[3])

		x4 = self.fc1(x3)
		x4 = self.fc2(x4)
		x4  =self.fc3(x4)

		x4 = x4.reshape(-1,self.output_dim,self.output_dim)

		return x4


	def setInput(self, data):
		file_path = PATH + 'running_model.pth'
		if os.path.isfile(file_path):
			self.load_state_dict(torch.load(file_path))

		# with open(file_name, 'r') as file:
		# 	data = file.readlines()

		data = data.splitlines()

		cnn_data = np.zeros((100, 26), dtype=float)
		lstm_data = np.zeros((100, 15), dtype=float)

		cnn_count = 0
		lstm_count = 0

		flag = 0
		for s in data:
			if s[:3] == "CNN":
				flag = 1
				continue

			elif s[:4] == "LSTM":
				flag = 2
				continue

			if flag == 1:
				s = s.replace('false','0')
				s = s.replace('true','1')
				s = s.replace('NaN','0')
				s = s.split()
				for i in range(len(s)):
					cnn_data[cnn_count][i] = float(s[i])
				cnn_count += 1

			elif flag == 2:
				s = s.replace('false','0')
				s = s.replace('true','1')
				s = s.replace('NaN','0')
				s = s.split()
				for i in range(len(s)):
					lstm_data[lstm_count][i] = float(s[i])
				lstm_count += 1

		self.vm_map = []
		for i in range(cnn_data.shape[0]):
			self.vm_map += [cnn_data[i][0]]

		file = open('vm_map.pickle','wb')
		pickle.dump(self.vm_map, file)

		train_cnn  = Variable(torch.from_numpy(cnn_data).type(torch.FloatTensor).view(1,cnn_data.shape[0],cnn_data.shape[1]))
		train_lstm = Variable(torch.from_numpy(lstm_data).type(torch.FloatTensor).view(1,lstm_data.shape[0],lstm_data.shape[1]))

		self.output = self.forward(train_cnn, train_lstm)
		self.output = self.output.view(self.output.shape[1],self.output.shape[2])

		file = open('output.pickle','wb')
		pickle.dump(self.output, file)
		# print(self.output)


	def backprop(self, data):
		optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

		# with open(file_name, 'r') as file:
		# 	data = file.readlines()

		data = data.splitlines()

		loss_parameters = []
		for d in data:
			d = d.replace('false','0')
			d = d.replace('true','1')
			d = d.replace('NaN','0')
			d = d.split()
			loss_parameters += [float(d[1])]
		
		loss_value = loss_parameters[3]/1000000 + loss_parameters[7] + loss_parameters[8]
		loss_value = torch.Tensor(np.array(loss_value)).type(torch.FloatTensor)

		file = open('output.pickle','rb')
		self.output = pickle.load(file)

		loss = self.output.min()
		loss.data = loss_value

		loss.backward()

		#update parameters
		optimizer.step()

		torch.save(model.state_dict(), PATH + 'running_model.pth')

		return str(loss.item())

	def host_rank(self, vm):
		vm = int(vm)	
		# print(self.output.shape)	

		file = open('output.pickle','rb')
		self.output = pickle.load(file)

		host_list = self.output.data[vm]
		# print(host_list)
		indices = np.flip(np.argsort(host_list))

	def migratableVMs(self):
		file = open('output.pickle','rb')
		self.output = pickle.load(file)

		file = open('vm_map.pickle','rb')
		self.vm_map = pickle.load(file)

		output_index = np.argmax(self.output.data, axis=1)
		
		migratableIndex = []
		for i in range(len(output_index)):
			if self.vm_map[i] != output_index[i].item():
				migratableIndex += [i]
		# print(migratableIndex)

	def sendMap(self, data):
		optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

		# with open(file_name, 'r') as file:
		# 	data = file.readlines()
		
		data = data.splitlines()

		vmMap = np.zeros((100,100), dtype=int)
		# print(vmMap.shape)

		file = open('output.pickle','rb')
		self.output = pickle.load(file)

		loss = 0
		for i in range(len(data)):
			l = data[i].split()
			y = int(l[1])
			loss += self.output[i][y]

		# print(loss.item())
		loss.backward()

		#update parameters
		optimizer.step()

		torch.save(model.state_dict(), PATH + 'running_model.pth')

		return str(loss.item())


if __name__ == '__main__':

	model = DeepRL(input_dim, hidden_dim, num_layers, output_dim, batch_size)

	# error = nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	file_name = "forward.txt"

	# print(model.fc1.weight)
	# model.setInput(file_name)

	file_name = "backprop.txt"

	model.backprop(file_name)
	# print(model.fc1.weight)

	vm = '2'
	model.host_rank(vm)

	model.migratableVMs()

	file_name = 'sendmap.txt'

	model.sendMap(file_name)

	# model.train(error, optimizer)

	# ind = 20
	# data_cnn = torch.from_numpy(np.array([features_test_cnn[ind]])).type(torch.FloatTensor)
	# data_lstm = torch.from_numpy(np.array([features_test_lstm[ind]])).type(torch.FloatTensor)
	# data_y = torch.from_numpy(np.array([targets_test[ind]])).type(torch.LongTensor)
	# # print(data_y.item())
	# test_data = torch.utils.data.TensorDataset(data_cnn,data_lstm,data_y)
	# cnn_data = Variable(test_data[0][0].view(1,test_data[0][0].shape[0],test_data[0][0].shape[1]))
	# lstm_data = Variable(test_data[0][1].view(1,test_data[0][1].shape[0],test_data[0][1].shape[1]))

	# output = model.test(cnn_data, lstm_data)
	# output = output.detach().numpy().reshape((10,10))
	# # predicted = torch.max(output.data, 1)
	# # print(predicted)

	# scheduler = DLScheduler()
	# vm = 9
	# print(output[vm])
	# indices = scheduler.host_rank(model, output, vm)
	# print(indices)
	