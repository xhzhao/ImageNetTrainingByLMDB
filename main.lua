local paths = require 'paths'
local tunnel = require 'tunnel'
local lmdb = require 'lmdb' 
require 'nnlr'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')
-------------------------------------------------------
-- 0
torch.setdefaulttensortype('torch.FloatTensor')

--***************************************************--

--------------------------------------------------------
-- 1. file configuration
-------------------------

local config = paths.dofile('config.lua')

--****************************************************--

--------------------------------------------------------
-- 2. cmd configuration
-----------------------

local opts = paths.dofile('opts.lua')
cmd = opts.parse(arg)
if cmd.model == 'alexnet' then
   config.croppedSize = {3,227,227}
end

--****************************************************--

--------------------------------------------------------
-- 3. create Data configuration
--  (0) phase(0/1)          cmd     train/test 
--  (1) data source(0/1)    cmd     Jpeg/LMDB 
--  (2) data path           config  string                              
--  (3) compress flag       config  true/false
--  (4) cropped image size  config  eg.{3, 224, 224}
--  (5) batchSize           cmd     int:32
--  (6) prefetchSize        config  int:5
--  (6) Normalization       config  eg.{'simple', 118.380948, 61.896913}
--  (7) ImageNetClasses     config  eg.1000
--  (8) ValidationLabels    config  eg.
--  (9) ImageMinSide        config  eg.256                                  --Minimum side length of saved images
--  (10)donkey              config  eg.4
--  (11)access way(0/1)     config  Seq/Random
-----------------------------------------------------
dataConfig = {
    phase           = cmd.phase,
    dataSource      = cmd.dataSource,
    dataPath        = config.path,
    Compressed      = config.Compressed,
    croppedSize     = config.croppedSize,
    batchSize       = cmd.batchSize,
    prefetchSize    = config.prefetchSize,
    Normalization   = config.Normalization,
    ImageNetClasses = config.ImageNetClasses,
    ValLabels       = config.ValidationLabels,
    ImageMinSide    = config.ImageMinSide,
    donkey          = config.donkey,
    accessWay       = config.accessWay,   
    epochs          = cmd.nEpochs, 
    epochSize       = cmd.epochSize
}
--print(dataConfig)
--****************************************************--

--------------------------------------------------------
-- 4. create Model configuration
--  (0) phase(0/1)          cmd     train/test 
--  (1) net type            cmd     string
--  (2) manualSeed          cmd     int
--  (3) epochs              cmd     int:250
--  (4) epochSize           cmd     int:40000
--  (5) epochNumber         cmd     int:1
--  (6) batchSize           cmd     int:32
--  (7) learning rate       cmd     float
--  (8) momentum            cmd     float
--  (9) weightDecay         cmd     float
--  (10)GPU/CPU             cmd     1/0
--  (11)nGPU                config  number
--  (12)cache               config  path string
--  (13)backend             config  cudnn | nn
--  (14)retrain             config  path string
--  (15)optimstate          config  path string
-----------------------------------------------------

modelConfig = {
    model           = cmd.model,
    phase           = cmd.phase,
    netType         = cmd.netType,
    optimization    = cmd.optimization,
    mannualSeed     = cmd.mannualSeed,
    epochs          = cmd.nEpochs,
    epochSize       = cmd.epochSize,
    epochNumber     = cmd.epochNumber,
    batchSize       = cmd.batchSize,
    nClasses        = config.nClasses,
    croppedSize     = config.croppedSize,
    learningRate    = cmd.learningRate,
    momentum        = cmd.momentum,
    weightDecay     = cmd.weightDecay,
    GPU_CPU         = cmd.GPU_CPU,
    nGPU            = config.nGPU,
    modelsFolder    = config.modelsFolder,
    cache           = config.cache,
    backend         = config.backend,
    retrain         = config.retrain,
    newRegime 	    = cmd.newRegime,
    prefetchSize    = config.prefetchSize,
    optimState      = config.optimState
}
print(modelConfig)
--****************************************************--

--------------------------------------------------------
-- 5. read LMDB dataset
-----------------------------------------------------
paths.dofile('LMDBProvider.lua')
TrainDB = LMDBProvider(dataConfig)
dataConfig.phase='val'
TestDB = LMDBProvider(dataConfig)
--****************************************************--

--------------------------------------------------------
-- 6. read model
-----------------------------------------------------
paths.dofile('createModel.lua')
TrainModel = netOptim(modelConfig)
print(TrainModel.network)
print("create model done")
--****************************************************--


--------------------------------------------------------
-- 7. read producer
-----------------------------------------------------
producer = function(TrainDB, TestDB, DataTensor, LabelTensor, BQInfo, coroutineInfo)

    print('produce haha')    
    local threads = require 'threads'
    local mutex = threads.Mutex()
    local conditionS = threads.Condition(coroutineInfo[1])
    local conditionF = threads.Condition(coroutineInfo[2])

    torch.setdefaulttensortype('torch.FloatTensor')
    local DB = TrainDB
    local epochs = DB.config.epochs
    local batchsize = DB.config.batchSize
    local epochsize = DB.config.epochSize
    local itemNum = batchsize*epochsize
    for i = 1, epochs do
	

	fReadLMDBAndPushData = function(bTrain)
	local maxIteration = 0
	if bTrain then
		maxIteration = epochsize
		print("train data loadding... , maxIteration = ", maxIteration)
	else
		maxIteration = 50000/batchsize
		print("test data loadding... , maxIteration = ", maxIteration)
	end
        for j = 1, maxIteration do
	    local t1 = sys.clock()
--	    local t3 = nil
            -- Buffer queue should not be modified if we havn't make sure this operation is secure
            mutex:lock()
--            print(BQInfo)
	    local converValue = BQInfo[3]*DB.config.prefetchSize
            local storeRunner =  BQInfo[1] + converValue
            local headDis = storeRunner - BQInfo[2]
--            print('store', headDis) 
	    if (headDis < 0) then
                print('Fatal Error, storeRunner fell behind fetchRunner ')
		mutex:unlock()
            elseif(headDis > DB.config.prefetchSize) then
                print('Fatal Error, storeRunner has led ahead fetchRunner a whole circle')
 		mutex:unlock()
            elseif(headDis == DB.config.prefetchSize) then
                --print('Warning, waiting for fetch data')
		conditionF:wait(mutex)
	        t1 = sys.clock()
                torch.setnumthreads(2)
                DB:cacheSeqBatch(j, epochsize, BQInfo[1]-1, DataTensor, LabelTensor)
--                t3 = sys.clock()
                if(BQInfo[1] == DB.config.prefetchSize) then
                    BQInfo[1] = 1
                    BQInfo[3] = 1
                else
                    BQInfo[1] = BQInfo[1] + 1
                end

                mutex:unlock()
                conditionS:signal()
                
            else
	        t1 = sys.clock()
                torch.setnumthreads(2)
	        DB:cacheSeqBatch(j, epochsize, BQInfo[1]-1, DataTensor, LabelTensor)
--                t3 = sys.clock()
                if(BQInfo[1] == DB.config.prefetchSize) then
                    BQInfo[1] = 1
                    BQInfo[3] = 1
                else
                    BQInfo[1] = BQInfo[1] + 1
                end

                mutex:unlock()
                conditionS:signal()
	    end
            local t2 = sys.clock()
	    --print('cacheData', t2-t1) --, 'pure read', t3-t1)
--            printer('producer', __threadid, i, j)
        end --end of epochsize
	end --end 0f fReadLMDBAndPushData
	DB = TrainDB
    	DB:open()
        DB:shuffle(itemNum)
	fReadLMDBAndPushData(true)
    	DB:close()
	DB = TestDB
	DB:open()
	fReadLMDBAndPushData(false)
    	DB:close()
	--start to load test data with the same batchsize


    end --end of epochs

    

end

--****************************************************--

--------------------------------------------------------
-- 8. consumer
--------------------------------------------------------

consumer = function(model, DataTensor, LabelTensor, BQInfo, coroutineInfo)
    print('consume haha')

    local threads = require 'threads'
    local mutex = threads.Mutex() 
    local conditionS = threads.Condition(coroutineInfo[1])
    local conditionF = threads.Condition(coroutineInfo[2])

    torch.setdefaulttensortype('torch.FloatTensor')

    local epochs = model.config.epochs
    local batchSize = model.config.batchSize
    local epochSize = model.config.epochSize
            
    local batchData  -- = torch.Tensor(batchSize, croppedSize[1], croppedSize[2], croppedSize[3]) 
    local batchLabel -- = torch.Tensor(batchSize)
    
    local lastTick = nil
    local interval = nil
    local totalerr = nil
    for i =1, epochs do
	model:TrainOneEpoch(i, model, DataTensor, LabelTensor, BQInfo, coroutineInfo)
	model:TestModel(i,model, DataTensor, LabelTensor, BQInfo, coroutineInfo)

    end

end

--***************************************************--


--------------------------------------------------------
-- 9. initalize environment for threads
init_job = function()
    local lmdb = require 'lmdb'
    local path = require 'paths'
    local nn = require 'nn'
    path.dofile('LMDBProvider.lua')
    path.dofile('createModel.lua')
    require 'nnlr'
    require 'tunnel' 

end
--***************************************************--

---------------------------------------------------------------
-- 10. create variables shared by producer and consumer threads

vector = tunnel.Vector(dataConfig.prefetchSize)
--printer = tunnel.Printer()


DataBufferTensor = torch.Tensor(dataConfig.prefetchSize * dataConfig.batchSize, config.croppedSize[1], config.croppedSize[2], config.croppedSize[3])
LabelBufferTensor = torch.Tensor(dataConfig.prefetchSize * dataConfig.batchSize)

storeCondition = threads.Condition()
storeConditionID = storeCondition:id()
fetchCondition = threads.Condition()
fetchConditionID = fetchCondition:id()
BQInfo = torch.LongTensor({1, 1, 0})   -- store_runner/fetch_runner/headWholeFliag
coroutineInfo = torch.LongTensor({storeConditionID, fetchConditionID})


--create blocks
producer_block = tunnel.Block(1, init_job)
consumer_block = tunnel.Block(1, init_job)
producer_block:add(TrainDB, TestDB, DataBufferTensor, LabelBufferTensor, BQInfo, coroutineInfo)
consumer_block:add(TrainModel, DataBufferTensor, LabelBufferTensor, BQInfo, coroutineInfo)
-- run threads
producer_block:run(producer)
consumer_block:run(consumer)

--**********************************************************--








