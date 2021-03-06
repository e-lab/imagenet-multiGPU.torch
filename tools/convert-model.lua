torch.setdefaulttensortype('torch.FloatTensor')

-- torch generic lib
require('pl')
require('nn')
require('cunn')
require('cudnn')


local opt = lapp([[
Required parameters
   --src            (default './model.t7')      path for source model file
   --dst            (default './model_new.t7')  path for destination model file

(Optional) parameters
   --dropout        (default true)        remove dropout (=true)
   --batchnorm      (default true)        remove batch normalization (=true)
   --softmax        (default true)        switch logsoftmax to softmax (=true)
   --precision                            check numerical error (=true)
]])

-- does not switch to softmax when calculating numerical error
if opt.precision then
   opt.softmax = false
end


local function merge_model_parallel(y, x, i)
   local xsub    = x.modules[i].modules
   local xsubsub = x.modules[i].modules[1].modules

   for j = 1, #xsubsub do

      if xsubsub[j].__typename == 'nn.Linear' then
         local input = xsubsub[j].weight:size(2)
         local output = xsubsub[j].weight:size(1)
         y:add(nn.Linear(input, output*#xsub))

         -- concatenate distributed parameters over different models
         for k = 1, #xsub do
            local range = {output*(k-1)+1, output*k}
            y.modules[#y.modules].weight[{range,{}}]:copy(xsub[k].modules[j].weight)
            y.modules[#y.modules].bias[{range}]:copy(xsub[k].modules[j].bias)
         end

      elseif xsubsub[j].__typename == 'nn.BatchNormalization' then
         local output = xsubsub[j].running_mean:nElement()
         local eps = xsubsub[j].eps
         local momentum = xsubsub[j].momentum
         local affine = xsubsub[j].affine
         y:add(nn.BatchNormalization(output*#xsub, eps, momentum, affine))
         y.modules[#y.modules].train = false

         -- concatenate distributed parameters over different models
         for k = 1, #xsub do
            local range = {output*(k-1)+1, output*k}
            y.modules[#y.modules].running_mean[{range}]:copy(xsub[k].modules[j].running_mean)
            y.modules[#y.modules].running_var[{range}]:copy(xsub[k].modules[j].running_var)
            if affine then
               y.modules[#y.modules].weight[{range}]:copy(xsub[k].modules[j].weight)
               y.modules[#y.modules].bias[{range}]:copy(xsub[k].modules[j].bias)
            end
         end

      -- copy and paste dimension-free modules
      else
         y:add(x.modules[i].modules[1].modules[j])
      end
   end
end


local function remove_parallelism(y, x)
   local y = y or nn.Sequential()

   for i = 1, #x.modules do
      -- recursively apply remove_parallelism
      if x.modules[i].__typename == 'nn.Sequential' then
         remove_parallelism(y, x.modules[i])

      -- copy and paste modules assuming no special module inside
      elseif x.modules[i].__typename == 'nn.Parallel' then
         y:add(x.modules[i])

      -- recursively apply remove_parallelism to only 1st branch (fbcunn module)
      elseif x.modules[i].__typename == 'nn.DataParallel' then
         remove_parallelism(y, x.modules[i].modules[1])

      -- recursively apply remove_parallelism to only 1st branch (cunn module)
      elseif x.modules[i].__typename == 'nn.DataParallelTable' then
         remove_parallelism(y, x.modules[i].modules[1])

      -- concatenate multiple model into one
      elseif x.modules[i].__typename == 'nn.ModelParallel' then
         merge_model_parallel(y, x, i)

      -- check if concat is model parallel or not
      elseif x.modules[i].__typename == 'nn.Concat' then
         -- check if inception module
         local same_depth = 0
         for j = 1, #x.modules[i].modules-1 do
            if (#x.modules[i].modules[j].modules == #x.modules[i].modules[j+1].modules) then
               same_depth = same_depth + 1
            end
         end
         local inception = not (same_depth == (#x.modules[i].modules-1))

         -- treat as ModelParallel unless module is inception
         if inception then
            y:add(x.modules[i])
         else
            merge_model_parallel(y, x, i)
         end

      -- otherwise apply simple copy and paste
      else
         y:add(x.modules[i])
      end
   end

   collectgarbage()
   return y
end


local function switch_backend(x)
   return cudnn.convert(x, nn)
end


local function squash_model(x, state)
   local i = 1
   while (i <= #x.modules) do
      if x.modules[i].__typename == 'nn.Sequential' then
         squash_model(x.modules[i], state)
      elseif x.modules[i].__typename == 'nn.Parallel' then
         squash_model(x.modules[i], state)
      elseif x.modules[i].__typename == 'nn.Concat' then
         squash_model(x.modules[i], state)
      elseif x.modules[i].__typename == 'nn.ConcatTable' then
         squash_model(x.modules[i], state)
      else
         -- remove dropout
         if state.dropout then
            if x.modules[i].__typename == 'nn.Dropout' or
               x.modules[i].__typename == 'nn.SpatialDropout' then
               x:remove(i)
               i = i - 1
            end
         end

         local absorb_bn_conv = function (w, b, mean, invstd, affine, gamma, beta)
            w:cmul(invstd:view(w:size(1),1):repeatTensor(1,w:nElement()/w:size(1)))
            b:add(-mean):cmul(invstd)

            if affine then
               w:cmul(gamma:view(w:size(1),1):repeatTensor(1,w:nElement()/w:size(1)))
               b:cmul(gamma):add(beta)
            end
         end

         local absorb_bn_deconv = function (w, b, mean, invstd, affine, gamma, beta)
            w:cmul(invstd:view(b:size(1),1):repeatTensor(w:size(1),w:nElement()/w:size(1)/b:nElement()))
            b:add(-mean):cmul(invstd)

            if affine then
               w:cmul(gamma:view(b:size(1),1):repeatTensor(w:size(1),w:nElement()/w:size(1)/b:nElement()))
               b:cmul(gamma):add(beta)
            end
         end

         -- remove batch normalization
         if state.batchnorm then
            if x.modules[i].__typename == 'nn.SpatialBatchNormalization'  then
               if x.modules[i-1] and
                 (x.modules[i-1].__typename == 'nn.SpatialConvolution' or
                  x.modules[i-1].__typename == 'nn.SpatialConvolutionMM') then
                  if not x.modules[i-1].bias then
                     x.modules[i-1].bias = torch.zeros(x.modules[i-1].nOutputPlane)
                  end
                  absorb_bn_conv(x.modules[i-1].weight,
                                 x.modules[i-1].bias,
                                 x.modules[i].running_mean,
                                 x.modules[i].running_var:clone():sqrt():add(x.modules[i].eps):pow(-1),
                                 x.modules[i].affine,
                                 x.modules[i].weight,
                                 x.modules[i].bias)
                  x:remove(i)
                  i = i - 1
               elseif x.modules[i-1] and
                     (x.modules[i-1].__typename == 'nn.SpatialFullConvolution') then
                  absorb_bn_deconv(x.modules[i-1].weight,
                                   x.modules[i-1].bias,
                                   x.modules[i].running_mean,
                                   x.modules[i].running_var:clone():sqrt():add(x.modules[i].eps):pow(-1),
                                   x.modules[i].affine,
                                   x.modules[i].weight,
                                   x.modules[i].bias)
                  x:remove(i)
                  i = i - 1
               else
                  -- comment this line to convert enet networks: so the first batchnorm is kept!
                  assert(false, 'Convolution module must exist right before batch normalization layer')
               end
            elseif x.modules[i].__typename == 'nn.BatchNormalization' and x.modules[i].running_std then
               if x.modules[i-1] and
                 (x.modules[i-1].__typename == 'nn.Linear') then
                  absorb_bn_conv(x.modules[i-1].weight,
                                 x.modules[i-1].bias,
                                 x.modules[i].running_mean,
                                 x.modules[i].running_std,
                                 x.modules[i].affine,
                                 x.modules[i].weight,
                                 x.modules[i].bias)
                  x:remove(i)
                  i = i - 1
               else
                  assert(false, 'Convolution module must exist right before batch normalization layer')
               end
            elseif x.modules[i].__typename == 'nn.BatchNormalization' and not x.modules[i].running_std then
               if x.modules[i-1] and
                 (x.modules[i-1].__typename == 'nn.Linear') then
                  absorb_bn_conv(x.modules[i-1].weight,
                                 x.modules[i-1].bias,
                                 x.modules[i].running_mean,
                                 x.modules[i].running_var:clone():sqrt():add(x.modules[i].eps):pow(-1),
                                 x.modules[i].affine,
                                 x.modules[i].weight,
                                 x.modules[i].bias)
                  x:remove(i)
                  i = i - 1
               else
                  assert(false, 'Convolution module must exist right before batch normalization layer')
               end
            end
         end

         -- replace logsoftmax with softmax
         if state.softmax then
            if x.modules[i].__typename == 'nn.LogSoftMax' then
               x:remove(i)
               x:insert(nn.SoftMax(), i)
            end
         end
      end
      i = i + 1
   end

   collectgarbage()
   return x
end


local function check_numerical_error(m1, m2, h, w)
   local x  = torch.randn(2, 3, h or 224, w or 224):cuda()

   m1 = m1:cuda()
   m2 = m2:cuda()

   local y1 = m1:forward(x)
   local y2 = m2:forward(x)

   local err = (y1 - y2):abs()
   print('==> numerical error (mean/max):', err:mean(), err:max())
end


local function convert_model(arg)
   -- load model
   local x = torch.load(arg.src)
   x:evaluate()

   print('==> before conversion')
   print(x)


   -- (1) remove parallel module (clone)
   local y_serial = remove_parallelism(nil, x:clone())

   -- (2) remove cuda dependent obj (inplace)
   local y_system = switch_backend(y_serial:float())

   -- (3) remove unnecessary modules (inplace)
   local y_squash = squash_model(y_system, arg)


   -- save model
   torch.save(arg.dst, y_squash)
   print('==> after conversion')
   print(y_squash)

   -- check error if needed
   if arg.precision then
      check_numerical_error(x, y_squash, nil, nil)
   end
end


convert_model(opt)
