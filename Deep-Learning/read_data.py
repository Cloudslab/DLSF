import numpy as np
import pickle

file = open('DL.txt','r')

# s = file.readline()
# print('p',' ','p')
loss_parameters = []
loss_parameters_instance = []

cnn_parameters = []
cnn_parameters_instance = []

lstm_parameters = []
lstm_parameters_instance = []

count = 1
n = 0
flag = 0
for x in file:
	#removing \n from the end of each line
	x = x.rstrip('\n')
	if x == 'LOSS : ':
		if flag == 3:
			lstm_parameters += [np.array(lstm_parameters_instance)]
			lstm_parameters_instance = []
		flag = 1
		continue

	elif x[:3] == 'CNN':
		flag = 2
		continue

	elif x[:4] == 'LSTM':
		flag = 3
		cnn_parameters += [np.array(cnn_parameters_instance)]
		cnn_parameters_instance = []
		continue

	elif x[:10] == 'Experiment':
		if flag == 3:
			lstm_parameters += [np.array(lstm_parameters_instance)]
			lstm_parameters_instance = []
		break

	if flag == 1:
		if x[:11] == 'CurrentTime':
			y = float(x[12:])
			loss_parameters_instance += [y]

		elif x[:8] == 'LastTime':
			y = float(x[9:])
			loss_parameters_instance += [y]			

		elif x[:8] == 'TimeDiff':
			y = float(x[9:])
			loss_parameters_instance += [y]

		elif x[:11] == 'TotalEnergy':
			y = float(x[12:])
			loss_parameters_instance += [y]

		elif x[:10] == 'NumVsEnded':
			y = float(x[11:])
			loss_parameters_instance += [y]

		elif x[:19] == 'AverageResponseTime':
			y = float(x[20:])
			loss_parameters_instance += [y]

		elif x[:20] == 'AverageMigrationTime':
			y = float(x[21:])
			loss_parameters_instance += [y]

		elif x[:21] == 'AverageCompletionTime':
			y = float(x[22:])
			# print(y)
			loss_parameters_instance += [y]

		elif x[:9] == 'TotalCost':
			y = float(x[10:])
			loss_parameters_instance += [y]

		elif x[:10] == 'SLAOverall':
			y = float(x[11:])
			loss_parameters_instance += [y]
			loss_parameters += [loss_parameters_instance]
			loss_parameters_instance = []
			# flag = 0

	elif flag == 2:
		x = x.replace('false','0')
		x = x.replace('true','1')
		x = x.replace('NaN','0')
		y = x.split()
		for i in range(len(y)):
			y[i] = float(y[i])
		cnn_parameters_instance += [np.array(y)]
		# print(y)
		# break

	elif flag == 3:
		x = x.replace('false','0')
		x = x.replace('true','1')
		x = x.replace('NaN','0')
		y = x.split()
		for i in range(len(y)):
			y[i] = float(y[i])
		lstm_parameters_instance += [np.array(y)]

# print(type(lstm_parameters))

# print(len(lstm_parameters[0][0]))

for i in range(len(lstm_parameters)):
	lstm_parameters[i] = lstm_parameters[i][:100]

# for i in range(len(cnn_parameters)):
# 	cnn_parameters[i] = cnn_parameters[i][:100]

loss_parameters = np.array(loss_parameters)
cnn_parameters = np.array(cnn_parameters)
lstm_parameters = np.array(lstm_parameters)

# print(loss_parameters.shape)
# print(len(cnn_parameters[0][0]))
# print(len(lstm_parameters[0]))

# for data in cnn_parameters:
# 	print(len(data))

# print(len(lstm_parameters[0][-1]))
# print(cnn_parameters[0][-1])

# #converting nan to 0
# loss_parameters[np.isnan(loss_parameters)] = 0
# # lstm_parameters[np.isnan(lstm_parameters)] = 0

# # print(len(loss_parameters))

# # for i in range(len(lstm_parameters)):
# # 	lstm_parameters[i] = lstm_parameters[i][:100]

# # for val in loss_parameters:
# # 	print(len(val))

# # for i in range(len(cnn_parameters)):
# # 	cnn_parameters[i] = cnn_parameters[i][:100]

# # lstm_parameters = np.array(val) for val in lstm_parameters
# # print(lstm_parameters.shape)

# # for val in cnn_parameters:
# # 	print(len(val))
# # print(len(cnn_parameters))
# np.save('loss_parameters.npy', loss_parameters)
# np.save('cnn_parameters.npy', cnn_parameters)
# np.save('lstm_parameters.npy', lstm_parameters)

loss_min_max = np.zeros((len(loss_parameters[0]),2))

for i in range(len(loss_parameters[0])):
	minm = float('inf')
	maxm = loss_parameters[0][i]
	for j in range(len(loss_parameters)):
		if loss_parameters[j][i] < minm:
			minm = loss_parameters[j][i]
			loss_min_max[i][0] = minm

		if loss_parameters[j][i] > maxm and loss_parameters[j][i] != float('inf'):
			maxm = loss_parameters[j][i]
			loss_min_max[i][1] = maxm

# print(loss_min_max)

lstm_min_max = np.zeros((len(lstm_parameters[0][0]),2))

for i in range(len(lstm_parameters[0][0])):
	minm = float('inf')
	maxm = float('-inf')
	for j in range(len(lstm_parameters)):
		for k in range(len(lstm_parameters[0])):
			if lstm_parameters[j][k][i] < minm:
				minm = lstm_parameters[j][k][i]
				lstm_min_max[i][0] = minm

			if lstm_parameters[j][k][i] > maxm:
				maxm = lstm_parameters[j][k][i]
				lstm_min_max[i][1] = maxm

cnn_min_max = np.zeros((len(cnn_parameters[0][0]),2))

for i in range(len(cnn_parameters[0][0])):
	minm = float('inf')
	maxm = float('-inf')
	for j in range(len(cnn_parameters)):
		for k in range(len(cnn_parameters[0])):
			if cnn_parameters[j][k][i] < minm:
				minm = cnn_parameters[j][k][i]
				cnn_min_max[i][0] = minm

			if cnn_parameters[j][k][i] > maxm:
				maxm = cnn_parameters[j][k][i]
				cnn_min_max[i][1] = maxm

file = open('loss_min_max.pickle','wb')
pickle.dump(loss_min_max, file)

file = open('cnn_min_max.pickle','wb')
pickle.dump(cnn_min_max, file)

file = open('lstm_min_max.pickle','wb')
pickle.dump(lstm_min_max, file)

