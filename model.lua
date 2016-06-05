--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' and not (opt.lastLayer == 'res34' or opt.lastLayer == 'fine')then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua
elseif opt.retrain ~= 'none' and opt.lastLayer == 'fine' or opt.lastLayer == 'lastLayerOnly'then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model, last = loadDataParallelLastLayerOnly(opt.retrain, opt.nGPU) -- defined in util.lua
end

-- 2. Create Criterion
criterion = nn.ClassNLLCriterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
-- model = model:cuda()
criterion:cuda()

collectgarbage()
