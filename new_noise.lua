-- Denoise the test image set using pretrained weights from encoder.lua

package.path = package.path .. ";" .. "/projects/pcsri/PetaVision/OpenPV/parameterWrapper/?.lua";
local pv = require "PVModule";


-- VThresh and nf should match the values used for training
if #arg ~= 4 then
	 print ("Usage: lua " .. arg[0] .. " <VThresh> <num features> <dictionary file> <output root dir>"); 
	 return(1);
end

local V1_filename	= "/projects/pcsri/PetaVision/run/output_nf128/dense_V1_0.4/Checkpoints/NoiseLayer_A.pvp";
local loadCheckpointDir = "/projects/pcsri/PetaVision/run/output_nf128/dense_V1_0.4/Checkpoints/"; 

local VThresh          = tonumber(arg[1]); --The threshold, or lambda, of the network
local dictionarySize   = tonumber(arg[2]);   --Number of patches/elements in dictionary 
local dictionaryFile   = nil; --  arg[3];    --nil for initial weights, otherwise, specifies the weights file to load.
local basePath         = arg[4];
local NoiseStdDev      = 0.5; -- 0.5;    -- Added noise is Gaussian 

local nbatch           = 1;    --Number of images to process in parallel
local nxSize           = 32;    --CIFAR images are 32 x 32
local nySize           = 32;
local patchSize        = 8;
local stride           = 2;
local displayPeriod    = 4000;   --Number of timesteps to find sparse approximation
local numEpochs        = 1;     --Number of times to run through dataset
local numImages        = 1; --Total number of images in dataset
local stopTime         = math.ceil((numImages  * numEpochs) / nbatch) * displayPeriod;
local writeStep        = displayPeriod; 
local initialWriteTime = displayPeriod; 

local inputPath        = "../CIFAR/repeated_batch.txt";
local outputPath       = basePath .. "/noisy_V1_" .. VThresh .. "/";
local checkpointDir    = outputPath .. "/Checkpoints";
local checkpointPeriod = (displayPeriod * 100); -- How often to write checkpoints



local plasticityFlag   = false;     --Determines if we are learning our dictionary or holding it constant
local momentumTau      = 500;       --Weight momentum parameter. A single weight update will last for momentumTau timesteps.
local dWMax            = 0.05;      --The learning rate
local AMin             = 0;
local AMax             = infinity;
local AShift           = VThresh;  --This being equal to VThresh is a soft threshold
local VWidth           = 0; 
local timeConstantTau  = 100;   --The integration tau for sparse approximation
local weightInit       = 1.0;

-- Base table variable to store
local pvParameters = {

	 --Layers------------------------------------------------------------
	 --------------------------------------------------------------------   
	 column = {
			groupType = "HyPerCol";
			startTime                           = 0;
			dt                                  = 1;
			stopTime                            = stopTime;
			progressInterval                    = (displayPeriod * 10);
			writeProgressToErr                  = true;
			verifyWrites                        = false;
			outputPath                          = outputPath;
			printParamsFilename                 = "CIFAR_Tutorial.params";
			randomSeed                          = 1234567890;
			nx                                  = nxSize;
			ny                                  = nySize;
			nbatch                              = nbatch;
			checkpointWrite                     = false;
			lastCheckpointDir                   = checkpointDir; -- Disable checkpointing except on the final step
			checkpointIndexWidth                = -1; -- Automatically select width of index in checkpoint directory name
			deleteOlderCheckpoints              = false;
			suppressNonplasticCheckpoints       = false;
			initializeFromCheckpointDir         = loadCheckpointDir;
			errorOnNotANumber                   = false;
	 };

	 AdaptiveTimeScales = {
			groupType                           = "AdaptiveTimeScaleProbe";
			targetName                          = "V1EnergyProbe";
			message                             = nil;
			textOutputFlag                      = true;
			probeOutputFile                     = "AdaptiveTimeScales.txt";
			triggerLayerName                    = "Input";
			triggerOffset                       = 0;
			baseMax                             = 0.06150000;  -- Initial upper bound for timescale growth
			baseMin                             = 0.06;  -- Initial value for timescale growth
			tauFactor                           = 0.03;  -- Percent of tau used as growth target
			growthFactor                        = 0.025; -- Exponential growth factor. The smaller value between this and the above is chosen. 
			writeTimeScales                     = true;
			writeTimeScalesFieldnames           = false;
	 };

	 Input = {
			groupType                           = "ImageLayer";
			nxScale                             = 1;
			nyScale                             = 1;
			nf                                  = 3;
			phase                               = 0;
			mirrorBCflag                        = true;
			writeStep                           = writeStep;
			initialWriteTime                    = initialWriteTime;
			sparseLayer                         = false;
			updateGpu                           = false;
			dataType                            = nil;
			inputPath                           = inputPath;
			offsetAnchor                        = "tl";
			offsetX                             = 0;
			offsetY                             = 0;
			inverseFlag                         = false;
			normalizeLuminanceFlag              = true;
			normalizeStdDev                     = true;
			useInputBCflag                      = false;
			autoResizeFlag                      = false;
			displayPeriod                       = displayPeriod;
			batchMethod                         = "byImage";
			writeFrameToTimestamp               = true;
			resetToStartOnLoop                  = false;
	 };

	 NoiseLayer = {
	 		-- This layer calculates the difference between the reconstructed
	 		-- image and the input.
	 		groupType = "ConstantLayer";

	 		-- Scale and features match input layer
	 		nxScale                             = 1;
	 		nyScale                             = 1;
	 		nf                                  = 3;
	 		phase                               = 1;
	 		mirrorBCflag                        = false;
	 		valueBC                             = 0;

	 		-- If true, initialize from the latest checkpoint in the directory
	 		-- specified in HyperCol. If a checkpoint exists, this will be
	 		-- used instead of InitVType below.
	 		initializeFromCheckpointFlag        = false;

	 		-- Initialization method for membrane potential buffer
	 		-- ZeroV: Set V to 0
	 		-- ConstantV: Set V to a constant value
	 		--     Use valueV parameter to set the constant
	 		-- UniformRandomV: Set V to a uniform distribution
	 		-- GaussianRandomV: Set V to a gaussian distribution
	 		-- InitVFromFile: Set V using a pvp file
	 		--     Use Vfilename parameter to specify the pvp file
	 		InitVType                           = "InitVFromFile"; 
			Vfilename                           = V1_filename;
	 		-- Layer is updated every timestep
	 		--triggerLayerName                    = NULL"Input";
	 		-- seed                                = 123456789;
	 		writeStep                           = writeStep;
	 		initialWriteTime                    = initialWriteTime;
	 		sparseLayer                         = false;
	 		updateGpu                           = false;
	 		dataType                            = nil;
	 		 -- stdDev                              = NoiseStdDev;
	 		 -- originalLayerName                   = "Input";
	 };

	 InputError = {
			groupType                           = "HyPerLayer";
			nxScale                             = 1;
			nyScale                             = 1;
			nf                                  = 3;
			phase                               = 2;
			mirrorBCflag                        = false;
			valueBC                             = 0;
			initializeFromCheckpointFlag        = false;
			InitVType                           = "ZeroV";
			triggerLayerName                    = NULL;
			writeStep                           = writeStep;
			initialWriteTime                    = initialWriteTime;
			sparseLayer                         = false;
			updateGpu                           = false;
			dataType                            = nil;
	 };

	 DenoiseError = {
			groupType                           = "HyPerLayer";
			nxScale                             = 1;
			nyScale                             = 1;
			nf                                  = 3;
			phase                               = 6;
			mirrorBCflag                        = false;
			valueBC                             = 0;
 			initializeFromCheckpointFlag        = true;
			InitVType                           = "ZeroV";
			triggerLayerName                    = NULL;
			writeStep                           = writeStep;
			initialWriteTime                    = initialWriteTime;
			sparseLayer                         = false;
			updateGpu                           = false;
			dataType                            = nil;
	 };

	 V1 = {
			groupType                           = "HyPerLCALayer";
			nxScale                             = 1/stride;
			nyScale                             = 1/stride;
			nf                                  = dictionarySize;
			phase                               = 3;
			mirrorBCflag                        = false;
			valueBC                             = 0;
			initializeFromCheckpointFlag        = true;
			--Vfilename			    = V1_filename;
			--InitVType                           = "InitVFromFile";
			valueV                              = 0;
			-- triggerLayerName                    = "Input";
			-- triggerBehavior			    = "resetStateOnTrigger";
			-- triggerResetLayerName		    = "BlankV1";
			writeStep                           = writeStep;
			initialWriteTime                    = initialWriteTime;
			sparseLayer                         = false;
			updateGpu                           = true;
			dataType                            = nil;
			VThresh                             = VThresh;
			AMin                                = AMin;
			AMax                                = AMax;
			AShift                              = AShift;
			VWidth                              = VWidth;
			timeConstantTau                     = timeConstantTau;
			selfInteract                        = true;
			adaptiveTimeScaleProbe              = "AdaptiveTimeScales";
	 };

	 InputRecon = {
			groupType = "HyPerLayer";
			nxScale                             = 1;
			nyScale                             = 1;
			nf                                  = 3;
			phase                               = 4;
			mirrorBCflag                        = false;
			valueBC                             = 0;
			initializeFromCheckpointFlag        = true;
			InitVType                           = "ZeroV";
			triggerLayerName                    = NULL;
			writeStep                           = writeStep;
			initialWriteTime                    = initialWriteTime;
			sparseLayer                         = false;
			updateGpu                           = false;
			dataType                            = nil;
	 };


	 --Connections ------------------------------------------------------
	 --------------------------------------------------------------------



	 InputToError = {
			groupType = "RescaleConn";
			preLayerName                        = "NoiseLayer";
			postLayerName                       = "InputError";
			channelCode                         = 0;
			delay                               = {0.000000};
			scale                               = weightInit;
	 };

	 InputToDenoiseError = {
			groupType = "RescaleConn";
			preLayerName                        = "Input";
			postLayerName                       = "DenoiseError";
			channelCode                         = 0;
			delay                               = {0.000000};
			scale                               = weightInit;
	 };

	 ErrorToV1 = {
			groupType = "TransposeConn";
			preLayerName                        = "InputError";
			postLayerName                       = "V1";
			channelCode                         = 0;
			delay                               = {0.000000};
			convertRateToSpikeCount             = false;
			receiveGpu                          = true;
			updateGSynFromPostPerspective       = true;
			pvpatchAccumulateType               = "convolve";
			writeStep                           = -1;
			writeCompressedCheckpoints          = false;
			selfFlag                            = false;
			gpuGroupIdx                         = -1;
			originalConnName                    = "V1ToInputError";
	 };

	 V1ToInputError = {
			groupType = "MomentumConn";
			preLayerName                        = "V1";
			postLayerName                       = "InputError";
			channelCode                         = -1;
			delay                               = {0.000000};
			numAxonalArbors                     = 1;
			plasticityFlag                      = plasticityFlag;
			convertRateToSpikeCount             = false;
			receiveGpu                          = false; -- non-sparse -> non-sparse
			sharedWeights                       = true;
			weightInitType                      = "UniformRandomWeight";
			wMinInit                            = -1;
			wMaxInit                            = 1;
			sparseFraction                      = 0.9;
			minNNZ                              = 0;
			useListOfArborFiles                 = false;
			combineWeightFiles                  = false;
			initializeFromCheckpointFlag        = true;
			triggerLayerName                    = "Input";
			triggerOffset                       = 0;
			updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
			pvpatchAccumulateType               = "convolve";
			writeStep                           = writeStep;
			initialWriteTime                    = initialWriteTime;
			writeCompressedCheckpoints          = false;
			selfFlag                            = false;
			nxp                                 = patchSize;
			nyp                                 = patchSize;
			shrinkPatches                       = false;
			normalizeMethod                     = "normalizeL2";
			strength                            = 1;
			normalizeArborsIndividually         = false;
			normalizeOnInitialize               = true;
			normalizeOnWeightUpdate             = true;
			rMinX                               = 0;
			rMinY                               = 0;
			nonnegativeConstraintFlag           = false;
			normalize_cutoff                    = 0;
			normalizeFromPostPerspective        = false;
			minL2NormTolerated                  = 0;
			dWMax                               = dWMax; 
			useMask                             = false;
			momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
			momentumMethod                      = "viscosity";
			momentumDecay                       = 0;
	 }; 

	 V1ToRecon = {
			groupType = "CloneConn";
			preLayerName                        = "V1";
			postLayerName                       = "InputRecon";
			channelCode                         = 0;
			delay                               = {0.000000};
			convertRateToSpikeCount             = false;
			receiveGpu                          = false;
			updateGSynFromPostPerspective       = false;
			pvpatchAccumulateType               = "convolve";
			writeCompressedCheckpoints          = false;
			selfFlag                            = false;
			originalConnName                    = "V1ToInputError";
	 };

	 ReconToError = {
			groupType = "IdentConn";
			preLayerName                        = "InputRecon";
			postLayerName                       = "InputError";
			channelCode                         = 1;
			delay                               = {0.000000};
			initWeightsFile                     = nil;
	 };

	 ReconToDenoiseError = {
			groupType = "IdentConn";
			preLayerName                        = "InputRecon";
			postLayerName                       = "DenoiseError";
			channelCode                         = 1;
			delay                               = {0.000000};
			initWeightsFile                     = nil;
	 };

	 --Probes------------------------------------------------------------
	 --------------------------------------------------------------------

	 V1EnergyProbe = {
			groupType                           = "ColumnEnergyProbe";
			message                             = nil;
			textOutputFlag                      = true;
			probeOutputFile                     = "V1EnergyProbe.txt";
			triggerLayerName                    = nil;
			energyProbe                         = nil;
	 };

	 StatsProbe = {
			groupType				  = "StatsProbe";
			targetLayer			  = "V1";
			message				  = nil;
			textOutputFlag			  = true;
			probeOutputFile			  = "StatsProbe.txt";
			maskLayerName			  = nil;
	 };

	 DenoiseErrorL2Norm = {
			groupType				  = "L2NormProbe";
			targetLayer 			  = "DenoiseError";
			message				  = nil;
			textOutputFlag			  = true;
			probeOutputFile			  = "DenoiseErrorL2Norm.txt";
			maskLayerName			  = nil;
	 };

	 InputErrorL2NormEnergyProbe = {
			groupType                           = "L2NormProbe";
			targetLayer                         = "InputError";
			message                             = nil;
			textOutputFlag                      = false;
			probeOutputFile                     = "InputErrorL2NormEnergyProbe.txt";
			energyProbe                         = "V1EnergyProbe";
			coefficient                         = 0.5;
			maskLayerName                       = nil;
			exponent                            = 2;
	 };

	 V1L1NormEnergyProbe = {
			groupType = "L1NormProbe";
			targetLayer                         = "V1";
			message                             = nil;
			textOutputFlag                      = false;
			probeOutputFile                     = "V1L1NormEnergyProbe.txt";
			energyProbe                         = "V1EnergyProbe";
			coefficient                         = VThresh;
			maskLayerName                       = nil;
	 };

} --End of pvParameters

if dictionaryFile ~= nil then
	 pvParameters.V1ToInputError.weightInitType  = "FileWeight";
	 pvParameters.V1ToInputError.initWeightsFile = dictionaryFile;
end
pv.printConsole(pvParameters)
