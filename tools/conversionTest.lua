-- Convert network
io.write('Converting model...'); io.flush()
local cudaModelPath = '/media/HDD1/model-cache/results-5k10k/20151113-163408-alexnetowtbn,epochNumber=52,nDonkeys=8,nEpochs=100000000,nGPU=4,normalize=f/model_68.t7'
local floatModelPath = '/media/HDD1/model-cache/results-5k10k/model_68_float.t7'
os.execute(string.format([[
th convert-model.lua \
--src %s \
--dst %s > /dev/null 2>&1
]], cudaModelPath, floatModelPath))
print(' Done.')

require 'cunn'
require 'cudnn'
require 'image'

-- Loading models
io.write('Loading models...'); io.flush()
cudaModel = torch.load(cudaModelPath)
for i, module in ipairs(cudaModel.modules[1].modules) do
   cutorch.setDevice(cudaModel.modules[1].gpuAssignments[i])
   cudaModel.modules[1].modules[i] = module:float():cuda()
end
cutorch.setDevice(1)

floatModel = torch.load(floatModelPath)
print(' Done.')

-- Creating inputs
floatInp = torch.FloatTensor(8, 3, 224, 224)
for i = 1, 8 do
   floatInp[i] = image.scale(image.lena(), 224, 224):float()
end

cudaInp = floatInp:clone():cuda()

-- Forward
io.write('Processing inputs...'); io.flush()
cudaModel:forward(cudaInp)
floatModel:forward(floatInp)
print(' Done.')

-- Comparison
print(floatModel.modules[19])
print(floatModel.modules[19].output[1][{ {1,10} }]:view(1, -1))

print(cudaModel.modules[2].modules[4])
print(cudaModel.modules[2].modules[4].output[1][{ {1,10} }]:view(1, -1))
