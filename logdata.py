import os
import subprocess


f = open("Log_2019_12_12_13_24.txt", "r")
#print(os.getcwd())
if f.mode == "r":
    contents = f.readlines()

f.close()
dict = {}

for line in contents:
    if(line.startswith("Section")):
        split = line.split('_')
        if split[0] not in dict.keys():
            dict[split[0]] = {}
        if split[1] not in dict[split[0]].keys():
            dict[split[0]][split[1]] = 1
        else:
            dict[split[0]][split[1]] += 1


print(dict)

