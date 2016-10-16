
-- require 'nn'
-- local nClasses = 40
-- torch.setdefaulttensortype('torch.FloatTensor')

function createModel(nGPU)
   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   -- small version by Eugenio Culurciello, September 2016
   
   nn.SpatialMaxPooling:ceil() -- compatibility with Caffe and other frameworks!
   
   local features = nn.Sequential()
   features:add(nn.SpatialConvolution(3,64,7,7,3,3))       -- 224 -> 73
   features:add(nn.SpatialBatchNormalization(64,1e-3))
   features:add(nn.SpatialMaxPooling(2,2,2,2):ceil())                   -- 73 ->  36
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(64,192,5,5,2,2))       --  36 -> 14
   features:add(nn.SpatialBatchNormalization(192,1e-3))
   features:add(nn.SpatialMaxPooling(2,2,2,2):ceil())                   --  14 ->  7
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  7 ->  7
   features:add(nn.SpatialBatchNormalization(384,1e-3))
   features:add(nn.SpatialMaxPooling(2,2,2,2):ceil())                   --  14 ->  7
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  7 ->  7
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  7 ->  7
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialAveragePooling(5, 5, 1, 1, 0, 0))      --  7 ->  1
  
   features:cuda()
   features = makeDataParallel(features, nGPU) -- defined in util.lua

   local classifier = nn.Sequential()
   classifier:add(nn.View(256))

   classifier:add(nn.Linear(256, nClasses))
   classifier:add(nn.LogSoftMax())

   classifier:cuda()

   local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 256
   model.imageCrop = 256

   return model
end

-- test code:
-- local a = torch.FloatTensor(1,3,256,256)--:cuda() -- input image test
-- local b = model:forward(a) -- test network
-- print(b:size())
