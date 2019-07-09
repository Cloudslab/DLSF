import matplotlib.pyplot as plt
import itertools
import statistics
import pickle
plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False

def reduce(l):
	n = 12
	res = []
	for i in range(0, len(l), n):
		res.append(statistics.mean(l[i:i+n]))
	return res

PATH = "../Models/"

# Models = ['FCN-AvgE', 'FCN-Res', 'FCN-Mig', 'FCN-Cost', 'FCN-SLA']
# Labels = ['\u03B1=1', '\u03B2=1', '\u03B3=1', '\u03B4=1', '\u03B5=1']

Models = ['LR-MMT', 'LRR-MC', 'MAD-MMT', 'FCN2', 'FCN-LR-MMT']
Labels = ['LR-MMT', 'LRR-MC', 'MAD-MMT', 'RL', 'FCN']

rot = 15 if len(Models) > 4 else 0

ParamNames = ['Energy (each interval)', 'Energy (total)', 'Number of Completed VMs', 'Response Time (average)',\
	'Response Time (each interval)', 'Response Time (total)', 'Migration Time (average)', 'Migration Time  (each interval)',\
	'Migration Time (total)',	'Completion Time (average)', 'Completion Time  (each interval)', 'Completion Time (total)',\
	'Cost  (each interval)', 'Cost', 'SLA Violations  (each interval)', 'Total SLA Violations',\
	'VMs migrated (each interval)', 'VMs migrated in total']

Titles = ['Energy (each interval)', 'Energy (total)', 'Number of Completed Tasks', 'Response Time (average)',\
	'Response Time (each interval)', 'Response Time (total)', 'Migration Time (average)', 'Migration Time  (each interval)',\
	'Migration Time (total)',	'Completion Time (average)', 'Completion Time  (each interval)', 'Completion Time (total)',\
	'Cost  (each interval)', 'Cost', 'SLA Violations  (each interval)', 'Total SLA Violations',\
	'Tasks migrated (each interval)', 'Tasks migrated in total']

yLabels = ['Interval Energy (Watts)', 'Total Energy (Watts)', 'Number of completed tasks',\
	'Average Response Time (seconds)', 'Interval Response Time (seconds)', 'Total Response Time (seconds)',\
	'Average Migration Time (seconds)', 'Interval Migration Time (seconds)', 'Total Migration Time (seconds)',\
	'Average Completion Time (seconds)', 'Interval Completion Time (seconds)', 'Total Completion Time (seconds)',\
	'Interval Cost (US Dollar)', 'Total Cost (US Dollar)', 'Fraction of SLA Violations', 'Fraction of SLA Violations',\
	'Number of Task migrations', 'Number of Task migrations']

xLabel = 'Simulation Time (minutes)'

# Colors = ['red', 'blue', 'green', 'orange', 'pink', 'cyan']
Colors = ['dimgray', 'darkgray', 'black', 'silver', 'grey']

Hatch = ['//', '+', '\\', '.', '+']

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
ModelLabels = dict(zip(Models,Labels))
ModelHatches = dict(zip(Models, Hatch))
yLabelDict = dict(zip(ParamNames,yLabels))
TitleDict = dict(zip(ParamNames,Titles))

for model in Models:
	for param in ParamNames:
		file = open('Data/'+model+'/'+param+'.pickle', 'rb')
		l = []
		l = pickle.load(file)
		Params[param][model] = l
		file.close()

# Graphs

x = range(5,24*60,5)

for paramname in ParamNames:
	# plt.title(TitleDict[paramname])
	plt.xlabel(xLabel)
	plt.ylabel(yLabelDict[paramname])
	for model in Models:
		x1 = range(5,24*60,5) if len(Params[paramname][model]) == 287 else range(0,24*60,5)
		x1 = x1[0:len(Params[paramname][model])]
		plt.plot(x1, Params[paramname][model], color=ModelColors[model], linewidth=1, label=ModelLabels[model], alpha=0.7)
	plt.legend()
	plt.savefig(paramname+".pdf")
	plt.clf()

for paramname in ParamNames:
	# plt.title(TitleDict[paramname])
	plt.xlabel('Simulation Time (Hours)')
	plt.ylabel(yLabelDict[paramname])
	for model in Models:
		plt.plot(reduce(Params[paramname][model]), color=ModelColors[model], linewidth=1, label=ModelLabels[model], alpha=0.7)
	plt.legend()
	plt.savefig("Reduced-"+paramname+".pdf")
	plt.clf()


## Cost
paramname = 'Cost'
# plt.title(TitleDict[paramname])
plt.xlabel('Model')
plt.ylabel(yLabelDict[paramname])
values = []
for model in Models:
	values.append(Params[paramname][model][-1])
plt.ylim(min(values)-statistics.stdev(values), max(values)+statistics.stdev(values))
plt.bar(range(len(values)), values, align='center', color=Colors)
plt.xticks(range(len(values)), Labels, rotation=rot)
plt.savefig(paramname+"-Bar.pdf")
plt.clf()

## Number of Completed VMs
paramname = 'Number of Completed VMs'
# plt.title(TitleDict[paramname])
plt.xlabel('Model')
plt.ylabel(yLabelDict[paramname])
values = []; values2 = []; values3 = []
for model in Models:
	values3.append(sum(Params[paramname][model]))
	values.append(sum(Params[paramname][model])*Params['Total SLA Violations'][model][-1])
	values2.append(sum(Params[paramname][model])*(1-Params['Total SLA Violations'][model][-1]))
plt.ylim(0, max(values3)+10*statistics.stdev(values3))
p1 = plt.bar(range(len(values)), values2, align='center', color='dimgray')
p2 = plt.bar(range(len(values)), values, bottom=values2, align='center', color='silver')
plt.xticks(range(len(values)), Labels, rotation=rot)
plt.legend((p1[0], p2[0]), ('Completed in Expected Time', 'Exceeded Expected Time'))
plt.savefig(paramname+"-Bar.pdf")
plt.clf()

## Total Energy
paramname = 'Energy (total)'
# plt.title(TitleDict[paramname])
plt.xlabel('Model')
plt.ylabel(yLabelDict[paramname])
values = []
for model in Models:
	values.append(Params[paramname][model][-1])
plt.ylim(min(values)-statistics.stdev(values), max(values)+statistics.stdev(values))
plt.bar(range(len(values)), values, align='center', color=Colors)
plt.xticks(range(len(values)), Labels, rotation=rot)
plt.savefig(paramname+"-Bar.pdf")
plt.clf()

## Total SLA violations
paramname = 'Total SLA Violations'
# plt.title(TitleDict[paramname])
plt.xlabel('Model')
plt.ylabel(yLabelDict[paramname])
values = []
for model in Models:
	values.append(Params[paramname][model][-1])
plt.ylim(min(values)-statistics.stdev(values), max(values)+statistics.stdev(values))
plt.bar(range(len(values)), values, align='center', color=Colors)
plt.xticks(range(len(values)), Labels, rotation=rot)
plt.savefig(paramname+"-Bar.pdf")
plt.clf()

## VMs migrated
paramname = 'VMs migrated in total'
# plt.title(TitleDict[paramname])
plt.xlabel('Model')
plt.ylabel(yLabelDict[paramname])
values = []
for model in Models:
	values.append(Params[paramname][model][-1])
plt.ylim(min(values)-statistics.stdev(values), max(values)+statistics.stdev(values))
plt.bar(range(len(values)), values, align='center', color=Colors)
plt.xticks(range(len(values)), Labels, rotation=rot)
plt.savefig(paramname+"-Bar.pdf")
plt.clf()

## Average Completion Time
paramname = 'Average Task Completion Time for a task'
# plt.title(paramname)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Model')
plt.xlabel('Time (seconds)')
values = []
for model in Models:
	values.append(Params['Completion Time (total)'][model][-1]/sum(Params['Number of Completed VMs'][model]))
plt.xlim(min(values)-statistics.stdev(values), max(values)+statistics.stdev(values))
plt.barh(range(len(values)), values, align='center', color=Colors)
plt.yticks(range(len(values)), Labels)
plt.savefig(paramname+"-Bar.pdf")
plt.clf()

## Response Time
paramname = 'Response Time (total)'
# plt.title(paramname)
plt.ylabel('Model')
plt.xlabel('Time (seconds)')
values = []
for model in Models:
	values.append(Params[paramname][model][-1])
plt.xlim(min(values)-statistics.stdev(values), max(values)+statistics.stdev(values))
plt.barh(range(len(values)), values, align='center', color=Colors)
plt.yticks(range(len(values)), Labels)
plt.savefig(paramname+"-Bar.pdf")
plt.clf()

# Output.txt

print("\t\t\t", end = '')
for p in ['Energy (total)\t\t', 'Response Time (total)', 'Completion Time (total)', 'Cost\t\t\t\t', 'Migration Time (total)', 'Total SLA Violations', 'VMs migrated in total']:
	print(p, end='\t')
print()

for model in Models:
	print(model, end='\t\t')
	for paramname in ['Energy (total)', 'Response Time (total)', 'Completion Time (total)', 'Cost', 'Migration Time (total)', 'Total SLA Violations', 'VMs migrated in total']:
		print("{:.2e}".format(Params[paramname][model][-1]), end='\t\t\t\t')
	print()





