import os

rootannot = "./data/odden/Annotations/"
rootmoveto = "./data/odden/JPEGImages/"
rootmovefrom = "./data/odden/part1/"

filenames = sorted(os.listdir(rootannot))

for idx, file in enumerate(filenames):
    filename = file[:-4] + ".jpg"
    
    os.replace(rootmovefrom+filename, rootmoveto + filename)
