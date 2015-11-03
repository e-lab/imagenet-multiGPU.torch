torch.setdefaulttensortype('torch.FloatTensor')

require('pl')
require('nn')
require('cunn')
require('xlua')
local dir = require('pl.dir')
local path = require('pl.path')


local opt = lapp([[
   --src   (default 'model.net')   path for source
]])
local dirname, modelname = path.splitpath(opt.src)
local statname = string.gsub(string.gsub(modelname,'model','stat'),'net','t7')
local dmname = string.gsub(string.gsub(modelname,'model','dm'),'net','png')

local path_model = opt.src
local path_stat = path.join(dirname, statname)
local path_dm = path.join(dirname, dmname)


local dataset_path = 'elab-faces-loose.t7'
assert(paths.filep(dataset_path))
local dataset = torch.load(dataset_path)
local nb_samples = dataset.data:size(1)


print('==> load pretrained network')
local model = torch.load(path_model):float()
local stat = torch.load(path_stat)
for c = 1, 3 do
   dataset.data[{{},{c},{},{}}]:add(-stat.mean[c]):div(stat.std[c])
end


-- prepare output storage
local embeddingSize = 128
local embeddings = torch.Tensor(nb_samples, embeddingSize)


print('==> calculate embeddings')
local batchSize = 128
for i = 1, nb_samples, batchSize do
   xlua.progress(i, nb_samples)
   local j = math.min(i+batchSize-1, nb_samples)
   local src = model:forward(dataset.data[{{i,j},{},{},{}}])
   local dst = embeddings[{{i,j},{}}]
   dst:copy(src)
end
torch.save('embeddings-test.t7', embeddings)


print('==> compute distance matrix')
local cnt = 0
local matrix = torch.Tensor(nb_samples, nb_samples):zero()
for i = 1, nb_samples do
   for j = i+1, nb_samples do
      cnt = cnt + 1
      xlua.progress(cnt, nb_samples*(nb_samples-1)/2)
      matrix[i][j] = torch.dist(embeddings[i], embeddings[j])
      matrix[j][i] = matrix[i][j]
   end
end
torch.save('distance-matrix.t7', matrix)


print('==> convert the matrix to jetmap')
require('image')
local k = 2 --matrix:max()
--matrix = matrix:div(2):mul(255):add(1)
matrix = matrix:div(k):mul(255):add(1)

--[[
-- Odd labels red and even labels blue
for i = 1, dataset.index:size(1), 2 do
   for j = dataset.index[i][1], dataset.index[i][2] do
      matrix[j][j] = 256-32
   end
end
for i = 2, dataset.index:size(1), 2 do
   for j = dataset.index[i][1], dataset.index[i][2] do
      matrix[j][j] = 32
   end
en:
]]

local jetmap = image.y2jet(matrix)
image.save(path_dm, jetmap)
