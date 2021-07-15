clear;
clc;

imagepath = "/transfer_learning/data/dataStore/";

checkpointpath = '/transfer_learning/training/checkpoints';


%% training data

volReader = @(x) matRead(x);
volLoc = fullfile(imagepath,'images');
volds = imageDatastore(volLoc, ...
    'FileExtensions','.mat','ReadFcn',volReader);


lblReader = @(x) segRead(x);
lblLoc = fullfile(imagepath,'labels');
classNames = ["background","LV","myo","RV"];
pixelLabelID = [0 1 2 3];
pxds = pixelLabelDatastore(lblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',lblReader);


patchSize = [256 256 24];
patchPerImage = 1;
miniBatchSize = 1;
patchds = randomPatchExtractionDatastore(volds,pxds,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;

dsTrain = transform(patchds,@(patchIn)preProcessData(patchIn));



%% validation data

volLocVal = fullfile(imagepath,'imagesVal');
voldsVal = imageDatastore(volLocVal, ...
    'FileExtensions','.mat','ReadFcn',volReader);

lblLocVal = fullfile(imagepath,'labelsVal');
pxdsVal = pixelLabelDatastore(lblLocVal,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',lblReader);

dsVal = randomPatchExtractionDatastore(voldsVal,pxdsVal,patchSize, ...
    'PatchesPerImage',patchPerImage);
dsVal.MiniBatchSize = miniBatchSize;


dsVal = transform(dsVal,@(patchIn)preProcessData(patchIn));



%% networkq


load('/pre_training/model/pre_trained_UNET.mat');

options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'InitialLearnRate',0.002, ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateSchedule','piecewise', ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',200, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'CheckpointPath',checkpointpath, ...
    'MiniBatchSize',miniBatchSize);


[net,info] = trainNetwork(dsTrain,layerGraph(net),options);

modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');

save(['/transfer_learning/models/final_Unet_model_' modelDateTime  '.mat'],'net','info');
