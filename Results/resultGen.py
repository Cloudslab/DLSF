import matplotlib.pyplot as plt
import itertools

PATH = "../Models/"

Models = ['FCN-LR-MMT']

ParamNames = ['Interval Energy', 'Total Energy', 'Number of Completed VMs', 'Average Response Time',\
	'Interval Response Time', 'Total Response Time', 'Average Migration Time', 'Total Migration Time',\
	'Average Completion Time', 'Interval Completion Time', 'Total Completion Time',\
	'Interval Cost', 'Total Cost', 'Interval SLA Violations', 'Total SLA Violations',\
	'VMs migrated in Interval', 'Total VMs migrated']

Colors = ['red', 'blue', 'green']

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
	IE = []; TE = [0]; NVE = []; ART = []; IRT = []; TRT = [0]; AMT = []; IMT = []; TMT = [0]
	ACT = []; ICT = []; TCT = [0]; IC = []; TC = [0]; ISLA = []; TSLA = [0]; IVM = []; TVM = [0]

	print("Parsing model "+model)

	file = open(PATH+model+"/DL.txt", "r")
	while(True):
		line = file.readline()
		if not line:
			break
		if not "TotalEnergy" in line:
			continue
		# Energy
		val = float(line.split(" ")[1])
		IE.append(val)
		TE.append(TE[-1] + val)
		# Num Vms Ended
		line = file.readline()
		val = float(line.split(" ")[1])
		NVE.append(val)
		# Response Time
		line = file.readline()
		val = float(line.split(" ")[1])
		ART.append(val)
		IRT.append(val * NVE[-1])
		TRT.append(TRT[-1] + IRT[-1])
		# Migration Time
		line = file.readline()
		val = float(line.split(" ")[1])
		AMT.append(val)
		IMT.append(val * NVE[-1])
		TMT.append(TMT[-1] + IMT[-1])
		# Completion Time
		line = file.readline()
		val = float(line.split(" ")[1])
		ACT.append(val)
		ICT.append(val * NVE[-1])
		TCT.append(TMT[-1] + IMT[-1])
		# Cost
		line = file.readline()
		val = float(line.split(" ")[1])
		IC.append(val)
		TC.append(TC[-1] + val)
		# SLA
		line = file.readline()
		val = float(line.split(" ")[1])
		ISLA.append(val)
		TSLA.append(TC[-1] + val)
		# VM Migrations
		line = file.readline()
		val = float(line.split(" ")[1])
		IVM.append(val)
		TVM.append(TVM[-1] + val)

	IntervalEnergy[model] = IE
	TotalEnergy[model] = TE
	NumVmsEnded[model] = NVE
	AvgResponseTime[model] = ART
	IntervalResponseTime[model] = IRT
	TotalResponseTime[model] = TRT
	AvgMigrationTime[model] = AMT
	IntervalMigrationTime[model] = IMT
	TotalMigrationTime[model] = TMT
	AvgCompletionTime[model] = ACT
	IntervalCompletionTime[model] = ICT
	TotalCompletionTime[model] = TCT
	IntervalCost[model] = IC
	TotalCost[model] = TC
	IntervalSLA[model] = ISLA
	TotalSLA[model] = TSLA
	IntervalVmsMigrated[model] = IVM
	TotalVmsMigrated[model] = TVM

x = range(5,24*12,5)

for paramname in ParamNames:
	plt.title(paramname)
	for model in Models:
		plt.plot(Params[paramname][model], marker='.', color=ModelColors[model], linewidth=1, label=paramname)
	plt.legend()
	plt.savefig(paramname+".png")
	plt.clf()



