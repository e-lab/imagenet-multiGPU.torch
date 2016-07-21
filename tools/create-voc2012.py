# Made by Sangpil Kim
# 2016 June

from subprocess import call
import os.path

# 20 classes
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', \
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', \
           'sheep', 'sofa', 'train', 'tvmonitor'  ]

#Copy Image function
def readTxt(txtExt,targetFolder):
    for target in classes:
        train_txt ='./ImageSets/Main/' + target + txtExt
        print(train_txt)
        f = open(train_txt,'r')
        lines = f.readlines()
        for line  in lines:
            index = int(line.split(' ')[-1].rstrip('\n'))
            if index == 1:
                targetIdx = str(line.split(' ')[0])
                filename = targetIdx + '.jpg'
                print(filename)
                cp = call(['cp', "JPEGImages/"+ filename, 'voc2012/'+targetFolder + target])
        f.close()

# Create folders
if not os.path.isdir('./train') or not os.path.isdir('./val'):
    print("creating train and val file")
    for target in classes:
        make_file = call(['mkdir', '-p', 'voc2012/train/'+target])
        make_file = call(['mkdir', '-p', 'voc2012/val/'+target])
else:
    print('Already processed')

#Copy Images
readTxt('_train.txt','train/')
readTxt('_trainval.txt','train/')
readTxt('_val.txt','val/')
