th main.lua -nGPU 4 -nDonkeys 8 -cropSize 224 -batchSize 128 -nClasses 257\
            -data /media/SuperSSD/spk/257/2nd_257Ori_1st_257Bing\
            -cache /media/HDD1/spk/257/checkpoints/fine\
            -optimizer adam \
            -regimes adam \
            -netType rpLast\
            -epochSize 120\
            -nEpochs 200\
            -retrain /media/HDD1/spk/model/pretrain/model_19.t7\
            -imageSize 256\
            -rpLastlayer