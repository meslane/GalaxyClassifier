import shutil
import csv

filename = "training_solutions_rev1/training_solutions_rev1.csv"
datapath = "images_training_rev1/images_training_rev1"
inclusionThresh = 0.6

with open(filename, newline = '') as f:
    rows = csv.reader(f)
    
    fields = next(rows)
    
    for i in range(len(fields)):
        fields[i] = fields[i].replace("Class","")
    
    #print(fields[1])
    
    for row in rows:
        for i in range(1, len(fields)):
            if (float(row[i]) >= inclusionThresh):
                print("IMG#{} meets criteria for {}".format(row[0], fields[i]))
                shutil.copy("{}/{}.jpg".format(datapath, row[0]), "sorted/{}/{}/{}.jpg".format(int(float(fields[i])), fields[i], row[0]))