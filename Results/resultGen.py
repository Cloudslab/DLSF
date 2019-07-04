import matplotlib.pyplot as plt
import itertools

PATH = "../Models/"

Models = ['FCN-LR-MMT', 'LR-MMT']

ParamNames = ['Energy (each interval)', 'Energy (total)', 'Number of Completed VMs', 'Response Time (average)',\
	'Response Time (each interval)', 'Response Time (total)', 'Migration Time (average)', 'Migration Time  (each interval)',\
	'Migration Time (total)',	'Completion Time (average)', 'Completion Time  (each interval)', 'Completion Time (total)',\
	'Cost  (each interval)', 'Cost', 'SLA Violations  (each interval)', 'Total SLA Violations',\
	'VMs migrated (each interval)', 'VMs migrated in total']

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

def parseLine(line):
	res = line.strip().split(" ")[1]
	if('Inf' in res or 'NaN' in res):
		return 0
	return float(res)

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
		val = parseLine(line)
		IE.append(val)
		TE.append(TE[-1] + val)
		# Num Vms Ended
		line = file.readline()
		val = parseLine(line)
		NVE.append(val)
		# Response Time
		line = file.readline()
		val = parseLine(line)
		ART.append(val if val != 0 else 0.001)
		IRT.append(val * NVE[-1])
		TRT.append(TRT[-1] + IRT[-1])
		# Migration Time
		line = file.readline()
		val = parseLine(line)
		AMT.append(val)
		IMT.append(val * NVE[-1])
		TMT.append(TMT[-1] + IMT[-1])
		# Completion Time
		line = file.readline()
		val = parseLine(line)
		ACT.append(val if val != 0 else ACT[-1])
		ICT.append(val * NVE[-1] if val != 0 else ICT[-1])
		TCT.append(TCT[-1] + ICT[-1])
		# Cost
		line = file.readline()
		val = parseLine(line)
		IC.append(val)
		TC.append(TC[-1] + val)
		# SLA
		line = file.readline()
		val = parseLine(line)
		ISLA.append(val)
		TSLA.append(TSLA[-1] + val)
		# VM Migrations
		line = file.readline()
		val = parseLine(line)
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
		plt.plot(Params[paramname][model], marker='.', color=ModelColors[model], linewidth=1, label=model)
	plt.legend()
	plt.savefig(paramname+".png")
	plt.clf()



