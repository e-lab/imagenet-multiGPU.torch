local ffi=require 'ffi'

function makeDataParallel(model, nGPU)
   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
      cutorch.setDevice(opt.GPU)
   end
   return model
end

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(opt.GPU)
   newDPT:add(module:get(1), opt.GPU)
   return newDPT
end

function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model))
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadDataParallel(filename, nGPU)
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

function saveRNGState(filename, donkeys, num_donkeys)
   local state = {}
   state.cpu = torch.getRNGState()
   state.gpu = {}
   state.donkeys = {}
   for i = 1, cutorch.getDeviceCount() do
      state.gpu[i] = cutorch.getRNGState(i)
   end
   if num_donkeys > 0 then
      donkeys:synchronize()
      donkeys:specific(true)
      for i = 1, num_donkeys do
         donkeys:addjob(i,
            function()
               return __threadid, torch.getRNGState()
            end,
            function(idx, thread_state)
               state.donkeys[idx] = thread_state
            end
         )
      end
      donkeys:synchronize()
      donkeys:specific(false)
   end
   torch.save(filename, state)
end

function loadRNGState(filename, donkeys, num_donkeys)
   local state = torch.load(filename)
   assert(cutorch.getDeviceCount() == #state.gpu, "Mismatch between number of GPUs and GPU RNG states")
   assert(num_donkeys == #state.donkeys, "Mismatch in donkey number")
   torch.setRNGState(state.cpu)
   for i = 1, cutorch.getDeviceCount() do
       cutorch.setRNGState(state.gpu[i], i)
   end
   if num_donkeys > 0 then
      donkeys:synchronize()
      donkeys:specific(true)
      for i = 1, num_donkeys do
         donkeys:addjob(i,
            function()
               torch.setRNGState(state.donkeys[__threadid])
            end
         )
      end
      donkeys:synchronize()
      donkeys:specific(false)
   end
end

selectParamsForEpoch = {}

function noResnetParamsForEpoch(epoch)
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

function resnetParamsForEpoch(epoch)
   local decay = 0
   if epoch == 1 then
   learningRate =  0.1
   else
      decay = math.floor((epoch - 1) / 30)
      learningRate = learningRate * math.pow(0.1,decay)
   end
   return { learningRate=learningRate , weightDecay= math.floor((epoch - 1) / 30) }, true
end

function initSelectParamsForEpoch(epoch, opt)
   if opt.regimes == 'res' then
      selectParamsForEpoch.update =
      function (epoch)
         return resnetParamsForEpoch(epoch)
      end
   else
      selectParamsForEpoch.update =
      function (epoch)
         return noResnetParamsForEpoch(epoch)
      end
   end
end

function selectParamsForEpoch:init(epoch, opt)
   initSelectParamsForEpoch(epoch, opt)
end

return selectParamsForEpoch
