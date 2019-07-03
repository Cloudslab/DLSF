import matplotlib.pyplot as plt

PATH = "../Models/"

Models = ['FCN-LR-MMT']

IntervalEnergy = {}
TotalEnergy = {}
NumVmsEnded = {}
AvgResponseTime = {}
TotalResponseTime = {}
AvgMigrationTime = {}
TotalMigrationTime = {}
AvgCompletionTime = {}
TotalCompletionTime = {}
IntervalCost = {}
TotalCost = {}
IntervalSLA = {}
TotalSLA = {}
IntervalVmsMigrated = {}
TotalVmsMigrated = {}

for model in Models:
	IE = []; TE = []; NVE = []; ART = []; TRT = []; AMT = []; TMT = []
	ACT = []; TCT = []; IC = []; TC = []; ISLA = []; TSLA = []; IVM = []; TVM = []
	