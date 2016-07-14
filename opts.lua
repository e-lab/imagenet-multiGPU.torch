--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-cache', './imagenet/checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', './imagenet/imagenet_raw_images/256', 'Home of ImageNet dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | nn')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        2, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-normalize',    true, 'globally normalize samples during training')
    cmd:option('-imgExtInsensitive', false, 'load JPEGs and PNGs regardless the file name extension')
    cmd:option('-imageSize',         256,    'Smallest side of the resized image')
    cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',        1000, 'number of classes in the dataset')
    cmd:text('------------- Training options --------------------')
    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       10000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       128,   'mini-batch size (1 = pure stochastic)')
    cmd:option('-batchChunks',     1,     'Number of splits per batch (e.g. if 2, then evey batch will be processed in two half-sized forward passes)')
    cmd:option('-conf',              'Compute and save confusion matrices')
    cmd:option('-verboseConf',    false,  'Print on screen / file the confusion matrices')
    cmd:text('---------- Optimization options ----------------------')
    cmd:option('-optimizer',  'sgd', 'Optimizer to train | adam | sgd')
    cmd:option('-regimes','pow', 'Option to chose regimes res is for res-net | res | noRes | pow')
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    cmd:text('---------- Model options ----------------------------------')
    cmd:option('-netType',     'alexnetowtbn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:option('-rngState',    'none', 'provide path to RNG state to reload from')
    cmd:option('-lastLayer',   'none', 'provide option to fine tune or train on last layer only | fine | lastLayerOnly')
    cmd:option('-color',        true,  'Option for color image augmentation')
    cmd:text()

    local opt = cmd:parse(arg or {})

    -- reflect dataset name in cached directory
    opt.cache = opt.cache .. '-' .. paths.basename(opt.data)

    -- add time-stamp, commandline specified options
    local date = os.date('%Y')..os.date('%m')..os.date('%d')
    local time = os.date('%H')..os.date('%M')..os.date('%S')
    opt.save = paths.concat(opt.cache,
                            cmd:string(opt.netType, opt,
                                       {netType=true, retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    return opt
end

return M
