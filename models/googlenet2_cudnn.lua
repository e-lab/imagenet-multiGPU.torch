require 'cudnn'

local function inception(input_size, config)
   local concat = nn.Concat(2)
   if config[1][1] ~= 0 then
      local conv1 = nn.Sequential()
      conv1:add(cudnn.SpatialConvolution(input_size, config[1][1],1,1,1,1)):add(nn.SpatialBatchNormalization(config[1][1], 1e-3)):add(cudnn.ReLU(true))
      concat:add(conv1)
   end

   local conv3 = nn.Sequential()
   conv3:add(cudnn.SpatialConvolution(  input_size, config[2][1],1,1,1,1)):add(nn.SpatialBatchNormalization(config[2][1], 1e-3)):add(cudnn.ReLU(true))
   conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1)):add(nn.SpatialBatchNormalization(config[2][2], 1e-3)):add(cudnn.ReLU(true))
   concat:add(conv3)

   local conv3xx = nn.Sequential()
   conv3xx:add(cudnn.SpatialConvolution(  input_size, config[3][1],1,1,1,1)):add(nn.SpatialBatchNormalization(config[3][1], 1e-3)):add(cudnn.ReLU(true))
   conv3xx:add(cudnn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1)):add(nn.SpatialBatchNormalization(config[3][2], 1e-3)):add(cudnn.ReLU(true))
   conv3xx:add(cudnn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1)):add(nn.SpatialBatchNormalization(config[3][2], 1e-3)):add(cudnn.ReLU(true))
   concat:add(conv3xx)

   local pool = nn.Sequential()
   pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting cudnn R2 into fbcode
   if config[4][1] == 'max' then
      pool:add(cudnn.SpatialMaxPooling(3,3,1,1):ceil())
   elseif config[4][1] == 'avg' then
      pool:add(cudnn.SpatialAveragePooling(3,3,1,1):ceil())
   else
      error('Unknown pooling')
   end
   if config[4][2] ~= 0 then
      pool:add(cudnn.SpatialConvolution(input_size, config[4][2],1,1,1,1)):add(nn.SpatialBatchNormalization(config[4][2], 1e-3)):add(cudnn.ReLU(true))
   end
   concat:add(pool)

   return concat
end

function createModel(nGPU)
   local features = nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,64,7,7,2,2,3,3)):add(nn.SpatialBatchNormalization(64, 1e-3)):add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
   features:add(cudnn.SpatialConvolution(64,64,1,1)):add(nn.SpatialBatchNormalization(64, 1e-3)):add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(64,192,3,3,1,1,1,1)):add(nn.SpatialBatchNormalization(192, 1e-3)):add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
   features:add(inception( 192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}})) -- 3(a)
   features:add(inception( 256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}})) -- 3(b)
   features:add(inception( 320, {{  0},{128,160},{ 64, 96},{'max',  0}})) -- 3(c)
   features:add(cudnn.SpatialConvolution(576,576,2,2,2,2))
   features:add(inception( 576, {{224},{ 64, 96},{ 96,128},{'avg',128}})) -- 4(a)

   local main_branch1 = nn.Sequential()
   main_branch1:add(inception( 576, {{192},{ 96,128},{ 96,128},{'avg',128}})) -- 4(b)
   main_branch1:add(inception( 576, {{160},{128,160},{128,160},{'avg', 96}})) -- 4(c)
   main_branch1:add(inception( 576, {{ 96},{128,192},{160,192},{'avg', 96}})) -- 4(d)

   local main_branch2 = nn.Sequential()
   main_branch2:add(inception( 576, {{  0},{128,192},{192,256},{'max',  0}})) -- 4(e)
   main_branch2:add(cudnn.SpatialConvolution(1024,1024,2,2,2,2))
   main_branch2:add(inception(1024, {{2*352},{2*192,2*320},{2*160,2*224},{'avg',2*128}})) -- 5(a)
   main_branch2:add(inception(2*1024, {{4*352},{4*192,4*320},{4*192,4*224},{'max',4*128}})) -- 5(b)
   main_branch2:add(cudnn.SpatialAveragePooling(7,7,1,1))
   main_branch2:add(nn.View(4*1024):setNumInputDims(3))
   main_branch2:add(nn.Linear(4*1024,nClasses))
   main_branch2:add(nn.LogSoftMax())

   local aux_classifier1 = nn.Sequential()
   aux_classifier1:add(cudnn.SpatialAveragePooling(5,5,3,3):ceil())
   aux_classifier1:add(cudnn.SpatialConvolution(576,128,1,1,1,1))
   aux_classifier1:add(nn.SpatialBatchNormalization(128, 1e-3))
   aux_classifier1:add(nn.ReLU(true))
   aux_classifier1:add(nn.View(128*4*4):setNumInputDims(3))
   aux_classifier1:add(nn.Linear(128*4*4,2048))
   aux_classifier1:add(nn.ReLU(true))
   aux_classifier1:add(nn.Linear(2048,nClasses))
   aux_classifier1:add(nn.LogSoftMax())

   -- add auxillary classifier here (thanks to Christian Szegedy for the details)
   local aux_classifier2 = nn.Sequential()
   aux_classifier2:add(cudnn.SpatialAveragePooling(5,5,3,3):ceil())
   aux_classifier2:add(cudnn.SpatialConvolution(576,128,1,1,1,1))
   aux_classifier2:add(nn.SpatialBatchNormalization(128, 1e-3))
   aux_classifier2:add(cudnn.ReLU(true))
   aux_classifier2:add(nn.View(128*4*4):setNumInputDims(3))
   aux_classifier2:add(nn.Linear(128*4*4,2048))
   aux_classifier2:add(nn.ReLU(true))
   aux_classifier2:add(nn.Linear(2048,nClasses))
   aux_classifier2:add(nn.LogSoftMax())

   -- split main_branch1 output between main_branch2 and aux_classifier2
   local splitter2 = nn.Concat(2)
   splitter2:add(main_branch2):add(aux_classifier2)

   -- place splitter2 on top of main_branch1
   local main_after_aux_1 = nn.Sequential()
   main_after_aux_1:add(main_branch1):add(splitter2)

   -- place remaining modules and aux_classifier1 on top of features
   -- main_after_aux_1 has to go before the classifier, because first nClasses outputs should
   -- come from the main model
   local splitter1 = nn.Concat(2)
   splitter1:add(main_after_aux_1):add(aux_classifier1)
   local model = nn.Sequential():add(features):add(splitter1)

   model:cuda()
   model = makeDataParallel(model, nGPU) -- defined in util.lua
   model.imageSize = 256
   model.imageCrop = 224
   model.auxClassifiers = 2
   model.auxWeights = {0.3, 0.3}


   return model
end
