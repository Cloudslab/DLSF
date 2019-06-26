import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

train = pd.read_csv(r"train.csv",dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# targets_numpy = targets_numpy[:500]
# features_numpy = features_numpy[:500]

# print(targets_numpy.shape[0])
features_train = features_numpy[:int(0.8*targets_numpy.shape[0])]
targets_train = targets_numpy[:int(0.8*targets_numpy.shape[0])]
features_test = features_numpy[int(0.8*targets_numpy.shape[0]):]
targets_test = targets_numpy[int(0.8*targets_numpy.shape[0]):]
# print(targets_test.shape)

# print(features_train[0].shape)

vm_train = np.random.randint(10, size=(features_train.shape[0], 10))
vm_test = np.random.randint(10, size=(features_test.shape[0], 10))

# x = vm_train[0][0]
# y = np.zeros(10)
# y[x] = 1
# print(y)
# vm_list = [vm_train[i][0] for i in range(len(vm_train))]
# print(len(vm_list))
# print(len(features_train))

featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)

featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

vmTrain = torch.from_numpy(vm_train)
vmTest = torch.from_numpy(vm_test)

batch_size = 50
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

# plt.imshow(features_numpy[10].reshape(28,28))
# plt.axis("off")
# plt.show()

input_dim = 28
hidden_dim = 28
num_layers = 2
output_dim = 10

learning_rate = 0.1

seq_dim = 28  
loss_list = []
iteration_list = []
accuracy_list = []
PATH = './models/'

class RNNModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim, batch_size):
		super(RNNModel, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.batch_size = batch_size

		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
		# self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
		
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.fc0 = nn.Linear(2*output_dim, output_dim)
		
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv3 = nn.Conv2d(1, 10, kernel_size = 8)
		self.conv4 = nn.Conv2d(1, 1, kernel_size=2)
		self.conv5 = nn.Conv2d(2, 1, kernel_size=4)
		self.conv6 = nn.Conv2d(1, 5, kernel_size=2)
		self.conv7 = nn.Conv2d(5, 1, kernel_size=2)
		self.conv8 = nn.Conv2d(2, 3, kernel_size=2)
		self.conv9 = nn.Conv2d(3, 1, kernel_size=3)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 200)
		self.fc2 = nn.Linear(200, hidden_dim)

	def init_hidden(self):
		return (torch.zeros(self.num_layers,self.batch_size, self.hidden_dim),
				torch.zeros(self.num_layers,self.batch_size, self.hidden_dim))

	def forward(self, x1, x2):
		# h0 = Variable(torch.zeros(self.layer_dim, x1.size(0,), self.hidden_dim))
		# out1, hn = self.lstm(x1, h0)
		# print(hn.shape)
		# # print(out1[:, -1, :].shape)
		# out = self.fc(out1[:, -1, :])
		
		# h_lstm1, out1 = self.lstm1(x1)
		# print(len(out1))
		# print(h_lstm1.shape)
		# h_lstm2, _ = self.lstm2(h_lstm1)

		lstm_out, self.hidden = self.lstm(x1)
		# print(self.hidden[1].shape)
		# print(lstm_out.shape)
		x4 = lstm_out.reshape((lstm_out.shape[0],1,lstm_out.shape[1],lstm_out.shape[2]))
		# lstm_out1.reshape((lstm_out.shape[0],1,lstm_out.shape[1],lstm_out.shape[2]))
		x4 = self.conv6(x4)
		x4 = self.conv7(x4)
		x4 = F.relu(F.max_pool2d(x4,2))
		# print(x4.shape)
		
		# x4 = F.relu(F.max_pool2d(self.conv4(x4),2))
		
		# print(x4.shape)
		# x4 = x4.reshape(-1,10,10,10)
		# print(x4.shape)

		x2.resize_((x2.shape[0],1,28,28))
		# print(x2.shape)

		x3 = self.conv6(x2)
		x3 = self.conv7(x3)
		x3 = F.relu(F.max_pool2d(x3,2))

		# x3 = F.relu(F.max_pool2d(self.conv4(x2),2))
		
		# print(x3.shape)

		x5 = torch.cat((x3,x4),1)
		# print(x5.shape)
		# x5 = x5.reshape(-1,2,10,10,10)
		# print(x5.shape)
		x5 = self.conv8(x5)
		x5 = self.conv9(x5)
		# print(x5.shape)
		
		# x5 = F.relu(self.conv5(x5))
		
		# print(x5.shape)

		### shape of x5 : batch_size, no of vms, no of hosts
		x5 = x5.reshape((-1,10,10))
		# print(x5.shape)
		return x5

		# x = F.relu(F.max_pool2d(self.conv1(x2), 2))
		# # print(x.shape)
		# x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		# # print(x.shape)
		# x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
		# # print(x.shape)
		# x = F.relu(self.fc1(x))
		# # print(x.shape)
		# x = F.dropout(x, training=self.training)
		# out2 = self.fc2(x)
		# # print(out2)
		# # print(out2.shape)
		# # out2 = self.fc2(x)

		# out3 = torch.add(lstm_out[:, -1, :], out2)
		# # out4 = self.fc(out3)
		# # print(out4[0].shape)
		# # print(vm[0].shape)
		# # out5 = torch.cat((out4, vm),1)
		# out = self.fc(out3)
		# # print(out)
		# return out

	def custom_loss(self, outputs, labels):
		outputs = torch.sum(outputs, dim=1)
		loss = 0
		labels = torch.nn.functional.one_hot(labels, 10).float()
		# print(outputs)
		for i in range(outputs.shape[0]):
			# print(outputs[i].shape)
			# print(labels[i].shape)
			loss += torch.dot(outputs[i], labels[i])
		# print(loss)
		return loss

	def train(self, error, optimizer):
		self.hidden = self.init_hidden()
		# print(model.hidden.shape)
		count = 0
		for epoch in range(num_epochs):
			for i, (images, labels) in enumerate(train_loader):
				# for i in range(len(vms[0])):
				train  = Variable(images.view(-1, seq_dim, input_dim))
				labels = Variable(labels)
					
				# Clear gradients
				optimizer.zero_grad()
				
				# Forward propagation
				outputs = self.forward(train, train)
				# print(outputs)
				
				# Calculate softmax and cross entropy loss
				loss = 0
				for i in range(outputs.shape[1]):
					loss += error(outputs[:,i,:],labels)
					
				# loss = self.custom_loss(outputs, labels)

				# Calculating gradients
				loss.backward()
				
				# Update parameters
				optimizer.step()
					
				count += 1
				
				if count % 5 == 0:
					# Calculate Accuracy         
					correct = 0
					total = 0
					# Iterate through test dataset
					flag = 0
					image_check = []
					label_check = []
					for images, labels in test_loader:
						# for i in range(len(vms[0])):
						if flag == 0:
							image_check += [images]
							label_check += [labels]
							flag = 1

						images = Variable(images.view(-1, seq_dim, input_dim))
						
						# Forward propagation
						outputs = self.forward(images, images)
						
						# Get predictions from the maximum value
						predicted = torch.max(outputs[:,0,:].data, 1)[1]
						# print(predicted)
						
						# Total number of labels
						total += labels.size(0)
						
						correct += (predicted == labels).sum()
					
					accuracy = 100 * correct / float(total)
					
					# store loss and iteration
					loss_list.append(loss.data)
					iteration_list.append(count)
					accuracy_list.append(accuracy)
					# if count % 5 == 0:
					# torch.save(model.state_dict(), PATH + 'new_model_' + str(count) + '.pth')	
					print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))

	def test(self, img):
		# print(featuresTest[1][205])
		# model = TheModelClass(*args, **kwargs)
		self.load_state_dict(torch.load('models/new_model_135.pth'))

		# data_x = torch.from_numpy(np.array(featuresTest))
		# test_data = torch.utils.data.TensorDataset(data_x)
		# img = Variable(test_data[:][0].view(-1, seq_dim, input_dim))
		output = self.forward(img, img)
		# print(output)
		# print((predicted == targetsTest).sum())
		return output

class DLScheduler():
	def host_rank(self, model, output, vm):
		host_list = output[vm]
		# print(host_list)
		indices = np.flip(np.argsort(host_list))
		# print(indices)
		# indices = np.flip(indices)
		# print(indices)
		return list(indices)

if __name__ == '__main__':

	model = RNNModel(input_dim, hidden_dim, num_layers, output_dim, batch_size)

	error = nn.CrossEntropyLoss()

	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	model.train(error, optimizer)

	ind = 20
	data_x = torch.from_numpy(np.array([features_test[ind]]))
	data_y = torch.from_numpy(np.array([targets_test[ind]]))
	# print(data_y.item())
	test_data = torch.utils.data.TensorDataset(data_x,data_y)
	img = Variable(test_data[0][0].view(-1, seq_dim, input_dim))

	output = model.test(img)
	output = output.detach().numpy().reshape((10,10))
	print(output[2])
	# predicted = torch.max(output.data, 1)
	# print(predicted)

	scheduler = DLScheduler()
	vm = 2
	indices = scheduler.host_rank(model, output, vm)
	print(indices)
	