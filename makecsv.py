import csv
import os
import json

rootdir = '/home/tre3x/python/DeepfakeDetection/data/traindata_dfdc'
with open('./folds.csv', 'w') as f:
    writer = csv.writer(f)

    writer.writerow(['img_file', 'label', 'fold'])
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
            if file.endswith(".jpg"):
                num = file.split('_')[0]
                try:
                  with open('/home/tre3x/python/DeepfakeDetection/data/traindata_dfdc/Metadata/metadata_{}.json'.format(num), "r") as myfile:
                    data = json.load(myfile)
                    label = data[file.split('_')[1]+'.mp4']['label']
                    if label == 'FAKE':
                      writer.writerow([filepath, 1, 1])
                    if label == 'REAL':
                      writer.writerow([filepath, 0, 1])
                except:
                    print(file)
                    pass
