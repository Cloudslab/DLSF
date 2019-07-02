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
import math
import matplotlib.pyplot as plt
from sys import stdin, stdout
import io

torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)


batch_size = 5
input_dim = 15
hidden_dim = 26
num_layers = 2
output_dim = 100

learning_rate = 0.00001

seq_dim = 1 
PATH = './model1/'

no_of_hosts = 100
no_of_vms = 100


class DeepRL(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim, batch_size):
		super(DeepRL, self).__init__()

		file_path = PATH + 'running_model.pth'

		if not(os.path.isdir(PATH)):
			# print("here")
			os.mkdir(PATH)
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.output_dim = output_dim
		self.hidden = []
		self.iter = 1
		self.loss_backprop = []
		self.loss_map = []

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

		if os.path.isfile(file_path):
			self.load_state_dict(torch.load(file_path))
			
			file = open(PATH + 'hidden_state.pickle','rb')
			self.hidden = pickle.load(file)

			file = open(PATH + 'loss_backprop.pickle','rb')
			self.loss_backprop = pickle.load(file)

			file = open(PATH + 'loss_map.pickle','rb')
			self.loss_map = pickle.load(file)

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
		# print(x4)
		x4 = F.softmax(x4, dim=2)
		# print(x4[0][0])

		return x4


	def setInput(self, cnn_data, lstm_data):
		# for name, param in self.named_parameters():
		#     if param.requires_grad:
		#         print(name, param.data)
		self.vm_map = []
		for i in range(cnn_data.shape[0]):
			self.vm_map += [cnn_data[i][0]]

		# file = open(PATH + 'vm_map.pickle','wb')
		# pickle.dump(self.vm_map, file)

		lstm_data = preprocessing.normalize(lstm_data)
		cnn_data = preprocessing.normalize(cnn_data)

		train_cnn  = Variable(torch.from_numpy(cnn_data).type(torch.FloatTensor).view(1,cnn_data.shape[0],cnn_data.shape[1]))
		train_lstm = Variable(torch.from_numpy(lstm_data).type(torch.FloatTensor).view(1,lstm_data.shape[0],lstm_data.shape[1]))

		# print(train_lstm.shape)
		self.output = self.forward(train_cnn, train_lstm)
		self.output = self.output.view(self.output.shape[1],self.output.shape[2])

		file = open(PATH+"DLoutput.txt", "w+")
		file.write(str(self.output))
		file.close()

		plt.imshow(self.output.detach().numpy(),cmap='gray')
		plt.savefig(PATH + 'DLoutput.jpg')
		plt.close()
		# file = open(PATH + 'output.pickle','wb')
		# pickle.dump(self.output, file)
		# print(self.output)


	def backprop(self, loss_parameters):
		if self.iter == 1:
			return("Init Loss")
		optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
		
		loss_value = loss_parameters[3]/1000000 + loss_parameters[7] + loss_parameters[8]
		loss_value = torch.Tensor(np.array(loss_value)).type(torch.FloatTensor)

		# file = open('output.pickle','rb')
		# self.output = pickle.load(file)

		loss = self.output.min()
		loss.data = loss_value

		loss.backward()

		#update parameters
		optimizer.step()

		if self.iter%10 == 0: 
			torch.save(model.state_dict(), PATH + 'running_model.pth')
			
			file = open(PATH + 'hidden_state.pickle','wb')
			pickle.dump(self.hidden, file)

			file = open(PATH + 'loss_backprop.pickle','wb')
			pickle.dump(self.loss_backprop, file)

			file = open(PATH + 'loss_map.pickle','wb')
			pickle.dump(self.loss_map, file)

			plt.plot(self.loss_backprop)
			plt.savefig(PATH + 'loss_backprop.jpg')
			plt.close()

			plt.plot(self.loss_map)
			plt.savefig(PATH + 'loss_map.jpg')
			plt.close()


		self.iter += 1

		self.loss_backprop += [loss.item()]
		return str(loss.item())

	def host_rank(self, vm):
		# print(self.output.shape)	

		# file = open('output.pickle','rb')
		# self.output = pickle.load(file)

		host_list = self.output.data[vm]
		# print(host_list)
		indices = np.flip(np.argsort(host_list))
		# print(indices)
		s = ''
		for index in indices:
			s += str(index) + ' '
		return s

	def migratableVMs(self):
		# file = open('output.pickle','rb')
		# self.output = pickle.load(file)

		# file = open('vm_map.pickle','rb')
		# self.vm_map = pickle.load(file)

		output_index = np.argmax(self.output.data, axis=1)
		
		migratableIndex = []
		for i in range(len(output_index)):
			if self.vm_map[i] != output_index[i].item():
				migratableIndex += [i]
		# print(migratableIndex)
		s = ''
		for index in migratableIndex:
			s += str(index) + ' '
		return s

	def sendMap(self, data):
		if self.iter == 1:
			self.iter +=1
			return "Init Loss"
		optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

		vmMap = np.zeros((100,100), dtype=int)
		# print(vmMap.shape)

		# file = open('output.pickle','rb')
		# self.output = pickle.load(file)

		loss = 0
		for i in range(len(data)):
			# l = data[i].split()
			y = data[i][1]
			vmMap[i][y] = 1
			# print(self.output[i][y])
			loss -= torch.log(self.output[i][y])

		plt.imshow(vmMap,cmap='gray')
		plt.savefig(PATH + 'sendMap.jpg')
		plt.close()

		file = open(PATH+"sendMap.txt", "w+")
		file.write(str(vmMap))
		file.close()

		file = open(PATH+"DLmap.txt", "w+")
		for i in range(len(data)):
			host_list = self.output.data[i]
			index = np.flip(np.argsort(host_list))[0]
			file.write(str(i) + " " + str(index) + "\n")
		file.close()

		loss /= len(data)
		# print(loss)
		loss.backward()

		#update parameters
		optimizer.step()

		if self.iter%5 == 0: 
			torch.save(model.state_dict(), PATH + 'running_model.pth')
			
			file = open(PATH + 'hidden_state.pickle','wb')
			pickle.dump(self.hidden, file)

			file = open(PATH + 'loss_backprop.pickle','wb')
			pickle.dump(self.loss_backprop, file)

			file = open(PATH + 'loss_map.pickle','wb')
			pickle.dump(self.loss_map, file)

			# globalFile.writeline(str(len(self.loss_map)))
			# globalFile.flush()
			plt.plot(self.loss_backprop)
			plt.savefig(PATH + 'loss_backprop.jpg')
			plt.clf()

			plt.plot(self.loss_map)
			plt.savefig(PATH + 'loss_map.jpg')
			plt.clf()


		self.iter += 1
		
		self.loss_map += [loss.item()]
		return str(loss.item())


if __name__ == '__main__':

	model = DeepRL(input_dim, hidden_dim, num_layers, output_dim, batch_size)

	# inp = "backprop,CurrentTime 300.1;LastTime 0.0;TimeDiff 300.1;TotalEnergy 105358.10624075294;NumVsEnded 1.0;AverageResponseTime 0.0;AverageMigrationTime 0.0;TotalCost 0.3317772222222221;SLAOverall NaN"
	# inp = "setInput,CNN data;1 2 3;4 5 6;LSTM data;7 8 9;1 2 3"
	# inp = "host_rank,4"
	inp = "sendMap,1 0;2 0;3 1;4 2;5 2;6 3"
	# inp = 'migratableVMs,'
	inp = []
	globalFile = open(PATH+"logs.txt", "a")

	while(True):
		while(True):
			line = stdin.readline()
			if "END" in line:
				break
			inp.append(line)
		if inp[0] == 'exit':
			break
		funcName = inp[0]
		data = inp[1:]
		inp = []

		if 'setInput' in funcName:
			file = open(PATH+"DLinput.txt", "w+")
			file.writelines(data)
			file.close()
			flag = 0
			cnn_data = np.zeros((100, 26), dtype=float)
			lstm_data = np.zeros((100, 21), dtype=float)
			cnn_count = 0
			lstm_count = 0
			for val in data:
				val = val.replace('false','0')
				val = val.replace('true','1')
				val = val.replace('NaN','0')
				x = val.split(' ')
				if x[0] == 'CNN':
					flag = 1
					continue
				
				elif x[0] == "LSTM":
					flag = 2
					continue

				if flag == 1:
					for i in range(len(x)):
						cnn_data[cnn_count][i] = float(x[i])
					cnn_count += 1

				elif flag == 2:
					for i in range(len(x)):
						lstm_data[lstm_count][i] = float(x[i])
					lstm_count += 1

			model.setInput(cnn_data, lstm_data)

		elif funcName == 'backprop':
			loss_data = []
			for val in data:
				val = val.replace('false','0')
				val = val.replace('true','1')
				val = val.replace('NaN','0')
				# print(val)
				val = val.split()
				loss_data += [float(val[1])]

			stdout.write(model.backprop(loss_data))
			stdout.flush()

		elif 'getSortedHost' in funcName:
			vm = int(data[0])
			stdout.write(model.host_rank(vm))
			stdout.flush()

		elif 'getVmsToMigrate' in funcName:
			stdout.write(model.migratableVMs())
			stdout.flush()

		elif 'sendMap' in funcName:
			file = open(PATH+"DLsendMap.txt", "w+")
			file.writelines(data)
			file.close()
			hostVmMap = []
			for val in data:
				val = val.split()
				l = [int(val[0]), int(val[1])]
				hostVmMap += [l]

			stdout.write(model.sendMap(hostVmMap)+"\n")
			stdout.flush()
