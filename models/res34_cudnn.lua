--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

require 'nn'
require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

function createModel(nGPU)
   local depth = 34
   local shortcutType =  'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local features = nn.Sequential()
   -- Configurations for ResNet:
   --  num. residual blocks, num features, residual block function
   local cfg = {
      [18]  = {{2, 2, 2, 2}, 512, basicblock},
      [34]  = {{3, 4, 6, 3}, 512, basicblock},
      [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
      [101] = {{3, 4, 23, 3}, 2048, bottleneck},
      [152] = {{3, 8, 36, 3}, 2048, bottleneck},
   }

   assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
   local def, nFeatures, block = table.unpack(cfg[depth])
   iChannels = 64
   print(' | ResNet-' .. depth .. ' ImageNet')

   -- The ResNet ImageNet model
   features:add(Convolution(3,64,7,7,2,2,3,3))
   features:add(SBatchNorm(64))
   features:add(ReLU(true))
   features:add(Max(3,3,2,2,1,1))
   features:add(layer(block, 64, def[1]))
   features:add(layer(block, 128, def[2], 2))
   features:add(layer(block, 256, def[3], 2))
   features:add(layer(block, 512, def[4], 2))
   features:add(Avg(7, 7, 1, 1))
   features:cuda()

   features = makeDataParallel(features, nGPU) -- defined in util.lua

   local classifier = nn.Sequential()
   classifier:add(nn.View(nFeatures):setNumInputDims(3))
   classifier:add(nn.Linear(nFeatures, 1000))
   classifier:add(nn.LogSoftMax())

   classifier:cuda()

   local model= nn.Sequential():add(features):add(classifier)

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end

   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')

   for i =1,2 do
      for k,v in pairs(model:get(i):findModules('nn.Linear')) do
         v.bias:zero()
      end
      if opt.backend == 'cudnn' then
         model:get(i):apply(function(m)
            if m.setMode then m:setMode(1,1,1) end
         end)
      end
      model:get(i).gradInput = nil
   end
   print('=============================================================')
   print(model:get(1))
   print('=============================================================')

   return model
end

