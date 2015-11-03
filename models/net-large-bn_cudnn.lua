function createModel(nGPU)
   require 'cudnn'

   -- td-net-large batch-normalized
   local features = nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,48,9,9,4,4,2,2))       -- 224 -> 55
   features:add(nn.SpatialBatchNormalization(48,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(cudnn.SpatialConvolution(48,128,5,5,1,1,2,2))       --  27 -> 27
   features:add(nn.SpatialBatchNormalization(128,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(cudnn.SpatialConvolution(128,192,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(192,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(192,192,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(192,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(192,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
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
      s:add(nn.Linear(256*6*6, 1024/nGPU))
      s:add(nn.BatchNormalization(1024/nGPU,1e-3))
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
      s:add(nn.Linear(1024, 1024/nGPU))
      s:add(nn.BatchNormalization(1024/nGPU,1e-3))
      s:add(nn.ReLU())
      branch2:add(s)
   end
   classifier:add(branch2)
   classifier:add(nn.Linear(1024, nClasses))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential():add(features):add(classifier)

   return model
end
