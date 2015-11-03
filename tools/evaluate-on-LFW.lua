torch.setdefaulttensortype('torch.FloatTensor')

require('torch')
require('nn')
require('cunn')

local batchSize = 100

-- load trained model
local model = torch.load('model.net')
local stat = torch.load('stat.t7')
model = model:cuda()

-- load dataset and normalize
local val_set = 10
local data = torch.load('/media/elab/SSD/datasets/LFW/lfw-dataset-unrestricted-loose.t7')[val_set]
local label = data.label[{{},{1}}]:eq(data.label[{{},{2}}])
for i = 1, 2 do
   data.data[i] = data.data[i]:float():div(255)
   for c = 1, 3 do
      data.data[i][{{},{c},{},{}}]:add(-stat.mean[c]):div(stat.std[c])
   end
   collectgarbage()
end

local x1 = torch.CudaTensor(batchSize, 3, 224, 224)
local x2 = torch.CudaTensor(batchSize, 3, 224, 224)
local y = torch.CudaTensor(batchSize)

for threshold = 0.1, 2, 0.1 do
   local sum = 0
   for i = 1, data.label:size(1), batchSize do
      local range = math.min(data.label:size(1)-i+1, batchSize)
      x1:copy(data.data[1][{{i,i+range-1},{},{},{}}])
      x2:copy(data.data[2][{{i,i+range-1},{},{},{}}])
      y:copy(label[{{i,i+range-1}}])

      local y1 = model:forward(x1):clone()
      local y2 = model:forward(x2):clone()
      local dist = y1:add(-y2):pow(2):sum(2):sqrt()
      local y_hat = dist:lt(threshold) -- 1 for same 0 for diff
      local acc = y_hat:eq(y):sum()
      sum = sum + acc

      collectgarbage()
   end
   print('==> acc', sum/data.label:size(1)*100, threshold)
end
