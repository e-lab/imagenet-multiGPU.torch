-- E. Culurciello, A. Paszke
-- ENet version 14 == for ImageNet

function createModel(nGPU)
   require 'cudnn'

   local features = nn.Sequential()

   local ct = 0
   function _bottleneck(internal_scale, use_relu, asymetric, dilated, input, output, downsample)
      local internal = output / internal_scale
      local input_stride = downsample and 2 or 1

      local sum = nn.ConcatTable()

      local main = nn.Sequential()
      local other = nn.Sequential()
      sum:add(main):add(other)

      main:add(cudnn.SpatialConvolution(input, internal, input_stride, input_stride, input_stride, input_stride, 0, 0):noBias())
      main:add(nn.SpatialBatchNormalization(internal, 1e-3))
      if use_relu then main:add(nn.PReLU(internal)) end
      if not asymetric and not dilated then
         main:add(cudnn.SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
      elseif asymetric then
         local pad = (asymetric-1) / 2
         main:add(cudnn.SpatialConvolution(internal, internal, asymetric, 1, 1, 1, pad, 0):noBias())
         main:add(cudnn.SpatialConvolution(internal, internal, 1, asymetric, 1, 1, 0, pad))
      elseif dilated then
         main:add(nn.SpatialDilatedConvolution(internal, internal, 3, 3, 1, 1, dilated, dilated, dilated, dilated))
      else
         assert(false, 'You shouldn\'t be here')
      end
      main:add(nn.SpatialBatchNormalization(internal, 1e-3))
      if use_relu then main:add(nn.PReLU(internal)) end
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

   local _ = require 'moses'
   local bottleneck = _.bindn(_bottleneck, 4, true, false, false) -- basic bottleneck
   local ibottleneck = _.bindn(_bottleneck, 1, true, false, false) -- basic bottleneck


   local initial_block = nn.ConcatTable(2)
   initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
   initial_block:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(initial_block) -- 112x112
   features:add(nn.JoinTable(2)) -- can't use Concat, because SpatialConvolution needs contiguous gradOutput
   features:add(nn.SpatialBatchNormalization(16, 1e-3))
   features:add(nn.PReLU(16))

   -- 1st block
   features:add(ibottleneck(16, 64, true)) -- 56x56
   features:add(bottleneck(64, 128))
   features:add(bottleneck(128, 128))

   -- 2nd block
   features:add(bottleneck(128, 256, true)) -- 28x28
   features:add(bottleneck(256, 256))
   features:add(bottleneck(256, 256))

   -- 3rd block
   features:add(bottleneck(256, 512, true)) -- 14x14
   features:add(bottleneck(512, 512))
   features:add(bottleneck(512, 512))

   -- 4th block
   features:add(bottleneck(512, 1024, true)) -- 7x7
   features:add(bottleneck(1024, 1024))
   features:add(bottleneck(1024, 1024))


   -- global average pooling 1x1
   features:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
   features:cuda()

   features = makeDataParallel(features, nGPU) -- defined in util.lua

   --classifier
   local classifier = nn.Sequential()
   classifier:add(nn.View(1024))
   classifier:add(nn.Linear(1024, nClasses))
   classifier:add(nn.LogSoftMax())
   classifier:cuda()

   -- test code:
   --local a = torch.FloatTensor(1,3,224,224):cuda() -- input image test
   --local b = model:forward(a) -- test network
   --print(b:size())

   local model = nn.Sequential():add(features):add(classifier)

   return model
end

