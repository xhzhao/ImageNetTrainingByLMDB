local paths = require 'paths'
local optim = require 'optim'
local net_optm = torch.class('netOptim')
require 'threads'
function net_optm:__init(config)
-- network
    network = self:createNetwork(config)
    self.network = network:float()
    
    self.parameters, self.gradParameters = self.network:getParameters()
-- Criterion
    self.criterion = nn.ClassNLLCriterion()
-- optimState
    self.optimState = self:initOptimState(config)
-- Logger
--    fileName = config.path .. '.log'
--    self.Logger = optim.Logger(paths.concat(config.savePath,  fileName))


-- other parameters may be needed
    self.config = config
end

function net_optm:initOptimState(config)
    optimState = {
        learningRate = config.learningRate,
        learningRateDecay = 0.0,
        momentum = config.momentum,
        --dampening = 0.0,
        --weightDecay = opt.weightDecay
    }

    if config.optimState ~= 'none' then
        assert(paths.filep(config.optimState), 'File not found: ' .. config.optimState)
        print('Loading optimState from file: ' .. config.optimState)
        optimState = torch.load(config.optimState)
    end

    return optimState
end


function net_optm:createNetwork(config)
    local network

    if config.retrain ~= 'none' then
        assert(paths.filep(config.retrain), 'File not found: ' .. config.retrain)
        print('Loading model from file: ' .. config.retrain);
        network = loadDataParallel(config.retrain, config.nGPU) -- defined in util.lua
    else
        paths.dofile(config.modelsFolder  .. config.netType .. '.lua')
        print('=> Creating model from file: models/' .. config.netType .. '.lua')
        network = createModel(config.nGPU, config.nClasses) -- for the model creation code, check the models/ folder
        if config.backend == 'cudnn' then
            require 'cudnn'
            cudnn.convert(network, cudnn)
        elseif config.backend ~= 'nn' then
            error'Unsupported backend'
       end
    end

    return network
end


function net_optm:paramsForEpoch(epoch)
     
-- L earning rate annealing schedule. We will build a new optimizer for
-- e ach epoch.
--   
-- B y default we follow a known recipe for a 55-epoch training. If
-- t he learningRate command-line parameter has been specified, though,
-- w e trust the user is doing something manual, and will use her
-- e xact settings for all optimization.
--   
-- R eturn values:
--     diff to apply to optimState,
--     true IFF this is the first epoch of a new regime
     if self.config.learningRate ~= 0.0 then -- if manually specified
        return { }
     end
     local regimes
    if self.config.model == 'googlenet' then
    regimes = {
        -- start, end,    LR,   WD,
        {  1,      8,    1e-2,     2e-4  },
        {  9,     16,    0.0096,   2e-4, },
        { 17,     24,    0.009216,  2e-4},
        { 25,     32,    0.00884736,   2e-4  },
        { 33,     40,    0.008493466,   2e-4 },
        { 41,     48,    0.008153727,   2e-4 },
        { 49,     56,    0.007827578,   2e-4 },
        { 57,     64,    0.007514475,   2e-4 },
        { 65,     72,    0.007213896,   2e-4 },
        { 73,     80,    0.00692534,   2e-4 },
        { 81,     88,    0.006648326,   2e-4 },
        { 89,     96,    0.006382393,   2e-4 },
        { 97,     104,   0.006127098,   2e-4 },
        { 105,    112,   0.005882014,   2e-4 },
        { 113,    120,   0.005646733,   2e-4 },
        { 121,    128,   0.005420864,   2e-4 },
        { 129,    136,   0.005204029,   2e-4 },
        { 137,    144,   0.004995868,   2e-4 },
        { 145,    152,   0.004796033,   2e-4 },
        { 153,    160,   0.004604192,   2e-4 },
        { 161,    168,   0.004420024,   2e-4 },
        { 169,    176,   0.004243223,   2e-4 },
        { 177,    184,   0.004073494,   2e-4 },
        { 185,    192,   0.003910555,   2e-4 },
        { 193,    200,   0.003754132,   2e-4 },
        { 201,    208,   0.003603967,   2e-4 },
        { 209,    216,   0.003459808,   2e-4 },
        { 217,    224,   0.003321416,   2e-4 },
        { 225,    232,   0.003188559,   2e-4 },
        { 233,    240,   0.003061017,   2e-4 },
        { 241,    248,   0.002938576,   2e-4 },
        { 249,    250,   0.002821033,   2e-4 },
    }
    elseif self.config.model == 'alexnet' then
    regimes = {
        -- start, end,    LR,   WD,
        {  1,     20,   1e-2,   5e-4, },
        { 21,     40,   1e-3,   5e-4  },
        { 41,     60,   1e-4,   5e-4 },
        { 61,     80,   1e-5,   5e-4 },
        { 80,     90,   1e-6,   5e-4 },
    }
    end

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end


function net_optm:setTrainOptim(epoch)
-- use every Epoch
    local params, newRegime = self:paramsForEpoch(epoch)
    local baseLR = params.learningRate
    local baseWD = params.weightDecay
    local LRs, WDs = self.network:getOptimConfig(1, baseWD)
    print("net_optm:setTrainOptim, baseLR = ",baseLR)
    if self.config.newRegime then
        --if config.useNNlr then
            self.optimState = {
                learningRate = baseLR,
                learningRateDecay = 0.0,
                momentum = self.config.momentum,
                dampening = 0.0,
                weightDecays = WDs,
                learningRates = LRs,
            }
--[[        else
            self.optimState = {
                learningRate = baseLR,
                learningRateDecay = 0.0,
                momentum = self.opt.momentum,
                dampening = 0.0,
                weightDecay = baseWD,
            }
        end  ]]--
    end

end

function net_optm:trainBatch(inputsCPU, labelsCPU)
--use every batch    

    local inputs = inputsCPU
    local labels = labelsCPU
    local err, outputs, totalerr
    model = self.network
    nClasses = self.config.nClasses
    criterion = self.criterion
    parameters = self.parameters
    gradParameters = self.gradParameters


    feval = function(x)
        torch.setnumthreads(42) 
        model:zeroGradParameters()
        outputs = model:forward(inputs)
        local model_outputs = outputs:sub(1, -1, 1, nClasses)
        err = criterion:forward(model_outputs, labels)
        totalerr = err

        local gradOutputs = criterion:backward(model_outputs, labels)
        if model.auxClassifiers and model.auxClassifiers > 0 then
            local allGradOutputs = torch.Tensor():typeAs(gradOutputs):resizeAs(outputs)
            allGradOutputs:sub(1, -1, 1, nClasses):copy(gradOutputs)
            auxerr = {}

            for i=1, model.auxClassifiers do
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

        model:backward(inputs, gradOutputs)
        return totalerr, gradParameters
    end

    optim[self.config.optimization](feval, parameters, self.optimState)
    -- DataParallelTable's syncParameters
    if model.needsSync then
        model:syncParameters()
    end
    return totalerr 
end


local top1_center, top5_center, loss
function net_optm:testBatch(inputsCPU, labelsCPU)

    local inputs = inputsCPU
    local labels = labelsCPU
    local err, outputs, totalerr
    model = self.network
    criterion = self.criterion

   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   local pred = outputs:float()

   loss = loss + err

   local _, pred_sorted = pred:sort(2, true)
   for i=1,pred:size(1) do
      local g = labelsCPU[i]
      if pred_sorted[i][1] == g then top1_center = top1_center + 1 end
      if pred_sorted[i][1] == g or pred_sorted[i][2] == g or pred_sorted[i][3] == g or pred_sorted[i][4] == g or pred_sorted[i][5] == g  then top5_center = top5_center + 1 end
   end


end

function net_optm:TrainOneEpoch(epoch, model, DataTensor, LabelTensor, BQInfo, coroutineInfo)
    local threads = require 'threads'
    local sys = require 'sys'

    local mutex = threads.Mutex() 
    local conditionS = threads.Condition(coroutineInfo[1])
    local conditionF = threads.Condition(coroutineInfo[2])

    local epochs = model.config.epochs
    local batchSize = model.config.batchSize
    local epochSize = model.config.epochSize
            
    local batchData 
    local batchLabel 
    
    local lastTick = nil
    local interval = nil
    local totalerr = nil

    model:setTrainOptim(epoch)
    model.network:training()
        for j =1, epochSize do

            if(j%100 == 0) then
               curTick = sys.clock() 
               if(lastTick ~= nil) then
                  interval = curTick - lastTick
               end
               lastTick = curTick
               print('train 100 batch time = ', __threadid, j,  interval, ' sec')
            end
            
            local t1 = sys.clock()
            local t2 = nil
            mutex:lock()

            local converValue = BQInfo[3]*model.config.prefetchSize
            local storeRunner =  BQInfo[1] + converValue
            local headDis = storeRunner - BQInfo[2]
            if(headDis > model.config.prefetchSize) then
                print('Fatal Error, storeRunner has led ahead fetchRunner a whole circle')
                mutex:unlock()
            elseif(headDis < 0) then
                print('Fatal Error, storeRunner has fell behind fetchRunner')
                mutex:unlock()
            elseif(headDis == 0) then
                --print('Warning, waiting for store data')
                conditionS:wait(mutex)
                local index = BQInfo[2]-1
                batchData = DataTensor[{{index*batchSize+1, (index+1)*batchSize}, {}, {}, {}}]
                batchLabel = LabelTensor[{{index*batchSize+1, (index+1)*batchSize}}]
                
                t2 = sys.clock()
                torch.setnumthreads(42)
                totalerr = model:trainBatch(batchData, batchLabel)
                torch.setnumthreads(1)
                if(BQInfo[2] == model.config.prefetchSize) then
                    BQInfo[2] = 1
                    BQInfo[3] = 0
                else
                    BQInfo[2] = BQInfo[2] + 1
                end
                mutex:unlock()
                conditionF:signal()
 
             else
                local index = BQInfo[2]-1
         
                batchData = DataTensor[{{index*batchSize+1, (index+1)*batchSize}, {}, {}, {}}]
                batchLabel = LabelTensor[{{index*batchSize+1, (index+1)*batchSize}}]
	        t2 = sys.clock()
                torch.setnumthreads(42)
                totalerr = model:trainBatch(batchData, batchLabel)
                torch.setnumthreads(1)
                if(BQInfo[2] == model.config.prefetchSize) then
                    BQInfo[2] = 1
                    BQInfo[3] = 0
                else
                    BQInfo[2] = BQInfo[2] + 1
                end
                mutex:unlock()
                conditionF:signal()
             end 
             local t3 = sys.clock()

             print("epoch=",epoch,",iteration =",j ,", LR = ", model.optimState.learningRate,", loss = ", totalerr, 'fetchdata', t2-t1, 'traindata', t3-t2)

            --    mutex:unlock()
        end
--        model:clearState()
--        saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
--        torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)


end


function net_optm:TestModel(epoch, model, DataTensor, LabelTensor, BQInfo, coroutineInfo)
    local threads = require 'threads'
    local sys = require 'sys'
    local mutex = threads.Mutex() 
    local conditionS = threads.Condition(coroutineInfo[1])
    local conditionF = threads.Condition(coroutineInfo[2])


    local batchSize = model.config.batchSize
    local batchData 
    local batchLabel 

    local totalTestStart = sys.clock()
    top1_center = 0
    top5_center = 0
    loss = 0
    local nTest = 50000 --test image db size
    print("testing start, nTest = ",nTest, ", batchSize = ",batchSize)
    model.network:evaluate()

        for j =1, nTest/batchSize do
            if(j%100 == 0) then
               curTick = sys.clock() 
               if(lastTick ~= nil) then
                  interval = curTick - lastTick
               end
               lastTick = curTick
               print('test index = ', j, ', test 100 batch time = ',interval, ' sec')
            end
            
            mutex:lock()

            local converValue = BQInfo[3]*model.config.prefetchSize
            local storeRunner =  BQInfo[1] + converValue
            local headDis = storeRunner - BQInfo[2]
            if(headDis > model.config.prefetchSize) then
                print('Fatal Error, storeRunner has led ahead fetchRunner a whole circle')
                mutex:unlock()
            elseif(headDis < 0) then
                print('Fatal Error, storeRunner has fell behind fetchRunner')
                mutex:unlock()
            elseif(headDis == 0) then
                --print('Warning, waiting for store data')
                conditionS:wait(mutex)
                local index = BQInfo[2]-1
                batchData = DataTensor[{{index*batchSize+1, (index+1)*batchSize}, {}, {}, {}}]
                batchLabel = LabelTensor[{{index*batchSize+1, (index+1)*batchSize}}]
                
                torch.setnumthreads(42)
                model:testBatch(batchData, batchLabel)
                torch.setnumthreads(1)
                if(BQInfo[2] == model.config.prefetchSize) then
                    BQInfo[2] = 1
                    BQInfo[3] = 0
                else
                    BQInfo[2] = BQInfo[2] + 1
                end
                mutex:unlock()
                conditionF:signal()
 
             else
                local index = BQInfo[2]-1
         
                batchData = DataTensor[{{index*batchSize+1, (index+1)*batchSize}, {}, {}, {}}]
                batchLabel = LabelTensor[{{index*batchSize+1, (index+1)*batchSize}}]
                torch.setnumthreads(42)
                model:testBatch(batchData, batchLabel)
                torch.setnumthreads(1)
                if(BQInfo[2] == model.config.prefetchSize) then
                    BQInfo[2] = 1
                    BQInfo[3] = 0
                else
                    BQInfo[2] = BQInfo[2] + 1
                end
                mutex:unlock()
                conditionF:signal()
             end
	print("test index = ", j)
        end

   local totalTestEnd = sys.clock()
   top1_center = top1_center * 100 / nTest
   top5_center = top5_center * 100 / nTest
   loss = loss / (nTest/batchSize) -- because loss is calculated per batch
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'averageLOSS (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t '
                          .. 'accuracy [Center](%%):\t top-5 %.2f\t ',
                       epoch, (totalTestEnd - totalTestStart), loss, top1_center, top5_center))

   print('\n')

end

