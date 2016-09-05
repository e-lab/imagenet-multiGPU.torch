--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

if opt.rngState ~= 'none' then
    assert(paths.filep(opt.rngState), 'File not found: ' .. opt.rngState)
    print('Loading RNG state from file: ' .. opt.rngState)
    loadRNGState(opt.rngState, donkeys, opt.nDonkeys)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local paramsForEpoch, learningRate
if opt.regimes == 'res' then
   function paramsForEpoch(epoch)
      local decay = 0, lr
      if epoch == 1 then
         if opt.LR ~= 0 then
            lr =  opt.LR
         else
            lr = 0.1
         end
      else
         decay = math.floor((epoch - 1) / 30)
         lr= lr * math.pow(0.1,decay)
      end
      return { learningRate = lr , weightDecay= math.floor((epoch - 1) / 30) }, true
   end
elseif opt.regimes == 'pow' then
   function paramsForEpoch(epoch)
      if opt.LR ~= 0.0 and epoch == 1 then -- if manually specified
         learningRate = opt.LR
         return { }
      elseif opt.LR == 0.0 and epoch == 1 then
         learningRate = 1e-2
      elseif epoch > 8 then
         learningRate = learningRate
      else
         learningRate = learningRate  *  math.pow( 0.9, epoch - 1)
      end
      local regimes = {
         -- start, end,     WD,
         {  1,     18,   5e-4, },
         { 19,     29,   5e-4  },
         { 30,     43,   0 },
         { 44,     52,   0 },
         { 53,    1e8,   0 },
      }
      for _, row in ipairs(regimes) do
         if epoch >= row[1] and epoch <= row[2] then
             return { learningRate = learningRate, weightDecay=row[3] }, true
         end
      end
   end
else
   function paramsForEpoch(epoch)
       if opt.LR ~= 0.0 then -- if manually specified
           return { }
       end
       local regimes = {
           -- start, end,    LR,   WD,
           {  1,     18,   1e-2,   5e-4, },
           { 19,     29,   5e-3,   5e-4  },
           { 30,     43,   1e-3,   0 },
           { 44,     52,   5e-4,   0 },
           { 53,    1e8,   1e-4,   0 },
       }

       for _, row in ipairs(regimes) do
           if epoch >= row[1] and epoch <= row[2] then
               return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
           end
       end
   end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, top5_epoch, loss_epoch
trainConf = opt.conf and optim.ConfusionMatrix(classes) or nil

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   top5_epoch = 0
   loss_epoch = 0
   if trainConf then trainConf:zero() end
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels
            local ok = xpcall(function()
                                 inputs, labels = trainLoader:sample(opt.batchSize)
                              end, function()
                                 print("ERROR!")
                                 print(debug.traceback())
                              end);
            if not ok then
               return
            end
            -- check the error
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   top5_epoch = top5_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['% top5 accuracy (train set)'] = top5_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t top-5 %.2f',
                       epoch, tm:time().real, loss_epoch, top1_epoch, top5_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   saveRNGState(paths.concat(opt.save, 'rngState_' .. epoch .. '.t7'), donkeys, opt.nDonkeys) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()
local parameters, gradParameters
-- Get only lastlayer parameters if want to train lastLayer only
if opt.lastLayer == 'lastLayerOnly' then
   parameters, gradParameters = model.modules[#model.modules-1]:getParameters()
   print(parameters:size())
else
   parameters, gradParameters = model:getParameters()
end

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   if not inputsCPU then
      print("Loader error. Skipping batch.")
      return
   end

   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   local outputsCPU = torch.FloatTensor(opt.batchSize, nClasses)

   local err, auxerr, totalerr, outputs
   feval = function(x)
      outputs = model:forward(inputs)
      local model_outputs = outputs:sub(1, -1, 1, nClasses)
      err = criterion:forward(model_outputs, labels)
      totalerr = err
      local gradOutputs = criterion:backward(model_outputs, labels)

      if model.auxClassifiers and model.auxClassifiers > 0 then
         local allGradOutputs = torch.Tensor():typeAs(gradOutputs):resizeAs(outputs)
         allGradOutputs:sub(1, -1, 1, nClasses):copy(gradOutputs)
         auxerr = {}
         for i=1,model.auxClassifiers do
            local first = i * nClasses + 1
            local last = (i+1) * nClasses
            local classifier_outputs = outputs:sub(1, -1, first, last)
            auxerr[i] = criterion:forward(classifier_outputs, labels)
            totalerr = totalerr + auxerr[i] * model.auxWeights[i]
            local auxGradOutput = criterion:backward(classifier_outputs, labels) * model.auxWeights[i]
            allGradOutputs:sub(1, -1, first, last):copy(auxGradOutput)
         end
         gradOutputs = allGradOutputs
      end
      -- This division could go into chunkedFeval but is in here for performance reasons
      gradOutputs:mul(1.0 / opt.batchChunks)

      model:backward(inputs, gradOutputs)
      return totalerr, gradParameters
   end

   local chunkedInputsCPU, chunkedLabelsCPU
   local transferToGPU = function(from, to)
      chunkedInputsCPU = inputsCPU:sub(from, to)
      chunkedLabelsCPU = labelsCPU:sub(from, to)
      inputs:resize(chunkedInputsCPU:size()):copy(chunkedInputsCPU)
      labels:resize(chunkedLabelsCPU:size()):copy(chunkedLabelsCPU)
   end

   -- This simulates feval by dividing every batch into chunks and calling
   -- the original function
   local chunkedFeval = function(x)
      local chunk_size = math.floor(opt.batchSize / opt.batchChunks)
      local err_accumulator = 0
      for i=1,opt.batchChunks do
         local chunk_start = chunk_size * (i-1) + 1
         -- Take all remaining samples in the last iteration
         local chunk_end = i < opt.batchChunks and chunk_size * i or -1
         transferToGPU(chunk_start, chunk_end)
         local loss, _ = feval(x)
         err_accumulator = err_accumulator + loss
         outputsCPU:sub(chunk_start, chunk_end):copy(outputs:sub(1, -1, 1, nClasses))
      end
      totalerr = err_accumulator / opt.batchChunks
      return total_loss, gradParameters
   end

   model:zeroGradParameters()
   if opt.optimizer == 'sgd' then
      optim.sgd(chunkedFeval, parameters, optimState)
   elseif opt.optimizer == 'adam' then
      optim.adam(chunkedFeval, parameters, optimState)
   else
      error('Wrong opt.optimizer should be sgd or adam')
   end

   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end


   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + totalerr
   -- top-1 and top-5 error
   local top1 = 0
   local top5 = 0
   do
      local _,max_prediction = outputsCPU:max(2)
      for i=1,opt.batchSize do
         if max_prediction[i][1] == labelsCPU[i] then
            top1_epoch = top1_epoch + 1;
            top1 = top1 + 1
         end
      end
      top1 = top1 * 100 / opt.batchSize;
   end
   do
      local _, predictions = outputsCPU:float():sort(2,true)
      local correct = predictions:eq(
         labelsCPU:long():view(opt.batchSize, 1):expandAs(outputsCPU:long()))
      local len = math.min(5, correct:size(1))
      local sumCorrect = correct:narrow(2, 1, len):sum()
      if sumCorrect > 0 then
         top5_epoch = top5_epoch + sumCorrect
         top5 = top5 + sumCorrect
      end
      top5 = top5 * 100 / opt.batchSize;
   end
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f Top5-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, totalerr, top1, top5,
          optimState.learningRate, dataLoadingTime))

   if model.auxClassifiers and model.auxClassifiers > 0 then
      print(string.format('\t  main model: Err %.4f', err))
      for i=1,model.auxClassifiers do
         print(string.format('\tclassifier %d: Err %.4f (* %.1f = %.4f)', i, auxerr[i], model.auxWeights[i], auxerr[i] * model.auxWeights[i]))
      end
   end

   if trainConf then trainConf:batchAdd(outputsCPU, labelsCPU) end

   dataTimer:reset()
end
