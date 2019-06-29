import pickle
import sys

file = open(sys.argv[1], "rb")
x = pickle.load(file)

print(x)