
#For Pascal

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

tar -xvf VOCtrainval_11-May-2012.tar

mv separatePascalDataset.py ./VOCdevkit/VOC2012

python separatePascalDataset.py
```

create-voc2012.py will create

```
voc2012/
├── train
│   ├── class
│   ├── class
│   └── ...
└── val
    ├── class
    └── class
    └── ...
```
