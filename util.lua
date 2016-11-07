require 'cunn'
local ffi=require 'ffi'

function makeDataParallel(model)
   if opt.nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(opt.nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1,true,true)
      for i=1, opt.nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
   end
   cutorch.setDevice(opt.GPU)

   return model
end


local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1,true,true)
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
   if opt.backend == 'cudnn' then
      require 'cudnn'
   end
	function replaceLastlayer(model)
      print('Replacing Last layer !!!!!')
		--Add Last Layer
		local featuressize
      if torch.type(model) == 'nn.DataParallelTable' then model = model:get(1) end
		if model.modules[#model].__typename == 'nn.Linear' then
			featuressize = model.modules[#model].weight:size(2)
		else
			--If last layer is not linear then it should be Logistic Function
			model:remove(#model)
			featuressize = model.modules[#model].weight:size(2)
		end
		--model.modules[nlayers] = nil  -- deleting nn.Linear layer
		model:remove(#model)  -- deleting nn.Linear layer
		local linear = nn.Linear(featuressize, nClasses)
		model:add(linear:cuda()):add(nn.LogSoftMax():cuda())
		return model
	end
	--Load file
   local model = torch.load(filename)
	--Check if it's lastlayer replace or finetune
	if opt.rpLastlayer then model = replaceLastlayer(model) end
	--Make it parallel
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' and opt.netType ~= 'faceBook' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model
   elseif torch.type(model) == 'nn.Sequential' and opt.netType == 'faceBook' then
		model = makeDataParallel(model, nGPU)
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end
function splitModel(model)
      local nlayers = model.modules[2]:size()
      model.modules[2].modules[nlayers] = nil  -- deleting nn.LogSoftMax layer
      local featuressize = model.modules[2].modules[nlayers - 1].weight:size(2)
      print(featuressize)
      model.modules[2].modules[nlayers - 1] = nil --deleting nn.Liner
      local last = nn.Sequential()
      local linear = nn.Linear(featuressize, nClasses)
      last:add(linear:cuda()):add(nn.LogSoftMax())
      return model, last
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
