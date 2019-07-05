import matplotlib.pyplot as plt
import itertools
import pickle

PATH = "../Models/"

Models = ['LR-MMT', 'LRR-MC', 'MAD-MMT', 'FCN-LR-MMT']

ParamNames = ['Energy (each interval)', 'Energy (total)', 'Number of Completed VMs', 'Response Time (average)',\
	'Response Time (each interval)', 'Response Time (total)', 'Migration Time (average)', 'Migration Time  (each interval)',\
	'Migration Time (total)',	'Completion Time (average)', 'Completion Time  (each interval)', 'Completion Time (total)',\
	'Cost  (each interval)', 'Cost', 'SLA Violations  (each interval)', 'Total SLA Violations',\
	'VMs migrated (each interval)', 'VMs migrated in total']

Colors = ['red', 'blue', 'green', 'orange']

Params = {}

IntervalEnergy = {}
TotalEnergy = {}
NumVmsEnded = {}
AvgResponseTime = {}
IntervalResponseTime = {}
TotalResponseTime = {}
AvgMigrationTime = {}
IntervalMigrationTime = {}
TotalMigrationTime = {}
AvgCompletionTime = {}
IntervalCompletionTime = {}
TotalCompletionTime = {}
IntervalCost = {}
TotalCost = {}
IntervalSLA = {}
TotalSLA = {}
IntervalVmsMigrated = {}
TotalVmsMigrated = {}

ParamList = [IntervalEnergy, TotalEnergy, NumVmsEnded, AvgResponseTime, IntervalResponseTime, \
TotalResponseTime, AvgMigrationTime, IntervalMigrationTime, TotalMigrationTime, AvgCompletionTime, \
IntervalCompletionTime, TotalCompletionTime, IntervalCost, TotalCost, IntervalSLA, TotalSLA, \
IntervalVmsMigrated, TotalVmsMigrated]

Params = dict(zip(ParamNames,ParamList))
ModelColors = dict(zip(Models,Colors))

for model in Models:
	for param in ParamNames:
		file = open('Data/'+model+'/'+param+'.pickle', 'rb')
		l = []
		l = pickle.load(file)
		Params[param][model] = l
		file.close()


x = range(5,24*12,5)

for paramname in ParamNames:
	plt.title(paramname)
	for model in Models:
		plt.plot(Params[paramname][model], color=ModelColors[model], linewidth=1, label=model, alpha=0.7)
	plt.legend()
	plt.savefig(paramname+".png")
	plt.clf()

print("\t\t\t", end = '')
for p in ['Energy (total)\t\t', 'Response Time (total)', 'Completion Time (total)', 'Cost\t\t\t\t', 'Migration Time (total)', 'Total SLA Violations', 'VMs migrated in total']:
	print(p, end='\t')
print()

for model in Models:
	print(model, end='\t\t')
	for paramname in ['Energy (total)', 'Response Time (total)', 'Completion Time (total)', 'Cost', 'Migration Time (total)', 'Total SLA Violations', 'VMs migrated in total']:
		print("{:.2e}".format(Params[paramname][model][-1]), end='\t\t\t\t')
	print()





