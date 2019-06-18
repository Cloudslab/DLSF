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

batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain,vmTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest,vmTest)

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

# plt.imshow(features_numpy[10].reshape(28,28))
# plt.axis("off")
# plt.show()

class RNNModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(RNNModel, self).__init__()
		self.hidden_dim = hidden_dim
		self.layer_dim = layer_dim
		self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim,
							batch_first=True,
							nonlinearity='relu')
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.fc0 = nn.Linear(2*output_dim, output_dim)
		
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 200)
		self.fc2 = nn.Linear(200, hidden_dim)

	def forward(self, x1, x2, vm):
		h0 = Variable(torch.zeros(self.layer_dim, x1.size(0,), self.hidden_dim))
		out1, hn = self.rnn(x1, h0)
		# # print(out1[:, -1, :].shape)
		# out = self.fc(out1[:, -1, :])

		x2.resize_((x2.shape[0],1,28,28))
		# # print(x2.shape)

		x = F.relu(F.max_pool2d(self.conv1(x2), 2))
		# print(x)
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		out2 = self.fc2(x)
		# print(out2)
		# print(out2.shape)
		# out2 = self.fc2(x)

		out3 = torch.add(out1[:, -1, :],out2)
		out4 = self.fc(out3)
		# print(out4[0].shape)
		# print(vm[0].shape)
		out5 = torch.cat((out4, vm),1)
		out = self.fc0(out5)
		# print(out)
		return out

# batch_size = 100
# n_iters = 2500
# num_epochs = n_iters / (len(features_train) / batch_size)
# num_epochs = int(num_epochs)

input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

error = nn.CrossEntropyLoss()

learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

seq_dim = 28  
loss_list = []
iteration_list = []
accuracy_list = []
count = 0
PATH = './models/'

for epoch in range(num_epochs):
	for i, (images, labels, vms) in enumerate(train_loader):
		for i in range(len(vms[0])):
			train  = Variable(images.view(-1, seq_dim, input_dim))
			# print(train.shape)
			# train = Variable(images)
			# print(train.shape)
			labels = Variable(labels)

			vm_list = np.array([vms[j][i] for j in range(len(vms))])
			vm_onehot = np.zeros((len(vms),output_dim))
			vm_onehot[np.arange(len(vms)),vm_list] = 1
			vm_onehot = torch.from_numpy(vm_onehot).float()
			vm_onehot = Variable(vm_onehot)
			# print(vm_onehot)
			# print(vm_onehot.shape)
				
			# Clear gradients
			optimizer.zero_grad()
			
			# Forward propagation
			outputs = model.forward(train, train, vm_onehot)
			
			# Calculate softmax and cross entropy loss
			loss = error(outputs, labels)
			
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
			for images, labels, vms in test_loader:
				for i in range(len(vms[0])):
					if flag == 0:
						image_check += [images]
						label_check += [labels]
						flag = 1

					images = Variable(images.view(-1, seq_dim, input_dim))

					vm_list = np.array([vms[j][i] for j in range(len(vms))])
					vm_onehot = np.zeros((len(vms),output_dim))
					vm_onehot[np.arange(len(vms)),vm_list] = 1
					vm_onehot = torch.from_numpy(vm_onehot).float()
					vm_onehot = Variable(vm_onehot)
					
					# Forward propagation
					outputs = model.forward(images, images, vm_onehot)
					
					# Get predictions from the maximum value
					predicted = torch.max(outputs.data, 1)[1]
					# print(predicted)
					
					# Total number of labels
					total += labels.size(0)
					
					correct += (predicted == labels).sum()
			
			accuracy = 100 * correct / float(total)
			
			# store loss and iteration
			loss_list.append(loss.data)
			iteration_list.append(count)
			accuracy_list.append(accuracy)
			if count % 5 == 0:
				# ind = 130
				# images = image_check[0]
				# # print(type(images[1][205]))
				# # print(images[1].shape)
				# img = Variable(images[1].view(-1, seq_dim, input_dim))
				# output = model.forward(img, img)
				# predicted = torch.max(output.data, 1)[1]
				# print(predicted)
				# print(label_check[0][1])

				# data_x = torch.from_numpy(np.array([features_test[1]]))
				# data_y = torch.from_numpy(np.array([targets_test[1]]))
				# test_data = torch.utils.data.TensorDataset(data_x,data_y)
				# # print(test_data.shape)

				# # for i in range(len(images[1])):
				# # 	if images[1][i].item() != featuresTest[1][i].item():
				# # 		print('here')
				# # 		break

				# # images = featuresTest
				# # print(type(images[1][205]))
				# # print(images[1].shape)
				# img = Variable(test_data[0][0].view(-1, seq_dim, input_dim))
				# output = model.forward(img, img)
				# predicted = torch.max(output.data, 1)[1]
				# print(predicted)
				# # print(images[0])
				# print(targetsTest[1])
				# torch.save(model.state_dict(), PATH + 'cnn_model_' + str(count) + '.pth')
				# Print Loss
				print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))


if __name__ == '__main__':
	# print(featuresTest[1][205])
	# model = TheModelClass(*args, **kwargs)
	model.load_state_dict(torch.load('models/model_1000.pth'))

	# correct = 0
	# total = 0
	# # Iterate through test dataset
	# for images, labels in test_loader:
	# 	images = Variable(images.view(-1, seq_dim, input_dim))
		
	# 	# Forward propagation
	# 	outputs = model.forward(images, images)
		
	# 	# Get predictions from the maximum value
	# 	predicted = torch.max(outputs.data, 1)[1]
		
	# 	# Total number of labels
	# 	total += labels.size(0)
		
	# 	correct += (predicted == labels).sum()

	# accuracy = 100 * correct / float(total)
	# print(type(accuracy.item()))

	# print(featuresTest[0].shape)
	
	# count = 0
	# for ind in tqdm(range(len(featuresTest))):
	# 	# ind = 245
	# 	# for i in range(10):
	# 	# print(featuresTest[ind])
		
	# 	data_x = torch.from_numpy(np.array([features_test[ind]]))
	# 	data_y = torch.from_numpy(np.array([targets_test[ind]]))
	# 	test_data = torch.utils.data.TensorDataset(data_x,data_y)
	# 	img = Variable(test_data[0][0].view(-1, seq_dim, input_dim))

	# 	# data_x = torch.from_numpy(np.array(featuresTest[:100]))
	# 	# test_data = torch.utils.data.TensorDataset(data_x)
	# 	# img = Variable(test_data[:][0].view(-1, seq_dim, input_dim))
	# 	# print(img)

	# 	# images = featuresTest[:100]
	# 	# images = np.array(images)
	# 	# images = images.reshape(-1,seq_dim, input_dim)
	# 	# # print(images.shape)
	# 	# # images = images.reshape((-1,seq_dim, input_dim))
	# 	# data_x = torch.from_numpy(images)
	# 	# data_point = torch.utils.data.TensorDataset(data_x)
	# 	# data_point = data_point[:][0]
		
	# 	# img = Variable(data_point.view(-1, seq_dim, input_dim))
	# 	# # print(targetsTest[ind])
	# 	# # print(img.shape)
	# 	output = model.forward(img, img)
	# 	predicted = torch.max(output.data, 1)[1]
	# 	# # # print(output)
	# 	# print(predicted)
	# 	# print(targetsTest[ind])
	# 	if predicted == targetsTest[ind]:
	# 		count += 1
	# 	# print((predicted == targetsTest[:100]))

	# 	# maxm = 0
	# 	# index = 0
	# 	# for j in range(output.shape[1]):
	# 	# 	if output[0][j].item() > maxm:
	# 	# 		maxm = output[0][j].item()
	# 	# 		index = j+1
	# 	# print(index)
	# print(count)

	data_x = torch.from_numpy(np.array(featuresTest))
	test_data = torch.utils.data.TensorDataset(data_x)
	img = Variable(test_data[:][0].view(-1, seq_dim, input_dim))
	output = model.forward(img, img)
	predicted = torch.max(output.data, 1)[1]
	print((predicted == targetsTest).sum())
