
function createModel(nGPU)
   require 'cudnn'
   require 'ccn2'
   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local features = nn.Sequential()
   features:add(nn.Transpose({1,4},{1,3},{1,2}))
   features:add(ccn2.SpatialConvolution(3,64,11,4,0,1,4))       -- 224 -> 55
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   -- 55 ->  27
   features:add(ccn2.SpatialConvolution(64,192,5,1,2,1,3))       --  27 -> 27
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   --  27 ->  13
   features:add(ccn2.SpatialConvolution(192,384,3,1,1,1,3))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialConvolution(384,256,3,1,1,1,3))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialConvolution(256,256,3,1,1,1,3))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   -- 13 -> 6
   features:add(nn.Transpose({4,1},{4,2},{4,3}))
   if nGPU > 1 then
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local features_single = features
      features = nn.DataParallel(1)
      for i=1,nGPU do
         cutorch.withDevice(i, function()
                               features:add(features_single:clone())
         end)
      end
      features.gradInput = nil
   end

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))

   local branch1
   if nGPU == 1 then
      branch1 = nn.Concat(2)
   else
      branch1 = nn.ModelParallel(2)
   end
   for i=1,nGPU do
      local s = nn.Sequential()
      s:add(nn.Dropout(0.5))
      s:add(nn.Linear(256*6*6, 4096/nGPU))
      s:add(nn.ReLU())
      branch1:add(s)
   end
   classifier:add(branch1)
   local branch2
   if nGPU == 1 then
      branch2 = nn.Concat(2)
   else
      branch2 = nn.ModelParallel(2)
   end
   for i=1,nGPU do
      local s = nn.Sequential()
      s:add(nn.Dropout(0.5))
      s:add(nn.Linear(4096, 4096/nGPU))
      s:add(nn.ReLU())
      branch2:add(s)
   end
   classifier:add(branch2)
   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential():add(features):add(classifier)

   return model
end
