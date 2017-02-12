-- E. Culurciello, A. Paszke
-- ENet version 14 == for ImageNet
-- this enet version with 128 input size

function createModel(nGPU)
   require 'cudnn'
   -- require 'nn'
   -- cudnn = nn
   -- nClasses = 10
   -- torch.setdefaulttensortype('torch.FloatTensor')

   local features = nn.Sequential()

   local ct = 0
   function bottleneck(internal_scale, input, output, downsample)
      local internal = output / internal_scale
      local input_stride = downsample and 2 or 1

      local sum = nn.ConcatTable()

      local main = nn.Sequential()
      local other = nn.Sequential()
      sum:add(main):add(other)

      main:add(cudnn.SpatialConvolution(input, internal, input_stride, input_stride, input_stride, input_stride, 0, 0):noBias())
      main:add(nn.SpatialBatchNormalization(internal, 1e-3))
      main:add(nn.PReLU(internal))
      main:add(cudnn.SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
      main:add(nn.SpatialBatchNormalization(internal, 1e-3))
      main:add(nn.PReLU(internal))
      main:add(cudnn.SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0):noBias())
      main:add(nn.SpatialBatchNormalization(output, 1e-3))
      main:add(nn.SpatialDropout((ct < 5) and 0.01 or 0.1))
      ct = ct + 1

      other:add(nn.Identity())
      if downsample then
         other:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      end
      if input ~= output then
         other:add(nn.Padding(1, output-input, 3))
      end

      return nn.Sequential():add(sum):add(nn.CAddTable()):add(nn.PReLU(output))
   end


   local initial_block = nn.ConcatTable(2)
   initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
   initial_block:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(initial_block) -- 112x112
   features:add(nn.JoinTable(2)) -- can't use Concat, because SpatialConvolution needs contiguous gradOutput
   features:add(nn.SpatialBatchNormalization(16, 1e-3))
   features:add(nn.PReLU(16))

   -- 1st block
   features:add(bottleneck(1, 16, 64, true)) -- 56x56
   features:add(bottleneck(4, 64, 128, false))
   features:add(bottleneck(4, 128, 128, false))

   -- 2nd block
   features:add(bottleneck(4, 128, 256, true)) -- 28x28
   features:add(bottleneck(4, 256, 256, false))
   features:add(bottleneck(4, 256, 256, false))

   -- 3rd block
   features:add(bottleneck(4, 256, 512, true)) -- 14x14
   features:add(bottleneck(4, 512, 512, false))
   features:add(bottleneck(4, 512, 512, false))


   -- global average pooling 1x1
   features:add(cudnn.SpatialAveragePooling(8, 8, 1, 1, 0, 0))
   features:cuda()

   features = makeDataParallel(features, nGPU) -- defined in util.lua

   --classifier
   local classifier = nn.Sequential()
   classifier:add(nn.View(512))
   classifier:add(nn.Linear(512, nClasses))
   classifier:add(nn.LogSoftMax())
   classifier:cuda()

   local model = nn.Sequential():add(features):add(classifier)
   -- print(model)

      -- test code:
   -- local a = torch.FloatTensor(1,3,128,128)--:cuda() -- input image test
   -- local b = model:forward(a) -- test network
   -- print(b:size())

   return model
end

