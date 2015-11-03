function createModel(nGPU)
   require 'cudnn'

   local features = nn.Sequential()

   -- 1
   features:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
   features:add(nn.SpatialBatchNormalization(64, 1e-3))
   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
   features:add(cudnn.ReLU(true))

   -- 2
   features:add(cudnn.SpatialConvolution(64, 192, 1, 1))
   features:add(nn.SpatialBatchNormalization(192, 1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(192, 192, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(192, 1e-3))
   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2)) --, 1, 1))
   features:add(cudnn.ReLU(true))

   -- 3
   features:add(cudnn.SpatialConvolution(192, 192, 1, 1))
   features:add(nn.SpatialBatchNormalization(192, 1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(192, 384, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(384, 1e-3))
   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2)) --, 1, 1))
   features:add(cudnn.ReLU(true))

   -- 4
   features:add(cudnn.SpatialConvolution(384, 384, 1, 1))
   features:add(nn.SpatialBatchNormalization(384, 1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(cudnn.ReLU(true))

   -- 5
   features:add(cudnn.SpatialConvolution(256, 256, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(cudnn.ReLU(true))

   -- 6
   features:add(cudnn.SpatialConvolution(256, 256, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
   features:add(nn.SpatialBatchNormalization(256, 1e-3))
   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2)) --, 1, 1))
   features:add(cudnn.ReLU(true))

   -- data parallel for convolutional part
   if nGPU > 1 then
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local features_single = features
      features = nn.DataParallel(1)
      for i=1,nGPU do
         cutorch.withDevice(i, function()
                               features:add(features_single:clone())
         end)
      end
   end

   -- model parallel for linear part
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))

   -- fc1
   local branch1
   if nGPU == 1 then
      branch1 = nn.Concat(2)
   else
      branch1 = nn.ModelParallel(2)
   end
   for i=1,nGPU do
      local s = nn.Sequential()
      s:add(nn.Linear(256*6*6, 4096/nGPU))
      s:add(nn.BatchNormalization(4096/nGPU, 1e-3))
      s:add(nn.ReLU())
      branch1:add(s)
   end
   classifier:add(branch1)

   -- fc2
   local branch2
   if nGPU == 1 then
      branch2 = nn.Concat(2)
   else
      branch2 = nn.ModelParallel(2)
   end
   for i=1,nGPU do
      local s = nn.Sequential()
      s:add(nn.Linear(4096, 4096/nGPU))
      s:add(nn.BatchNormalization(4096/nGPU, 1e-3))
      s:add(nn.ReLU())
      branch2:add(s)
   end
   classifier:add(branch2)

   -- fc128
   classifier:add(nn.Linear(4096, 128))
   classifier:add(nn.Normalize(2))

   local model = nn.Sequential():add(features):add(classifier)
   return model
end
