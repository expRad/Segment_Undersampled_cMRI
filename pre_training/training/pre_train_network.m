clear;
clc;

preprocessDataLocImages = "/pre_training/data/dataStoreKaggle";
preprocessDataLocLabels = "/pre_training/data/dataStoreKaggle";

% precompute coilsets
coilsets = precompute_Set_of_SENSis(12,1000);


%% training data

volReader = @(x) matRead(x);
volLoc = fullfile(preprocessDataLocImages,'images');
volds = imageDatastore(volLoc, ...
    'FileExtensions','.mat','ReadFcn',volReader);


lblReader = @(x) segRead(x);
lblLoc = fullfile(preprocessDataLocLabels,'labels');
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

dataSource = 'Training';
dsTrain = transform(patchds,@(patchIn)augmenter3Dcine_MC(patchIn,dataSource,squeeze(coilsets(randi(size(coilsets,1)),:,:,:))  ));



%% validation data

volLocVal = fullfile(preprocessDataLocImages,'imagesVal');
voldsVal = imageDatastore(volLocVal, ...
    'FileExtensions','.mat','ReadFcn',volReader);

lblLocVal = fullfile(preprocessDataLocLabels,'labelsVal');
pxdsVal = pixelLabelDatastore(lblLocVal,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',lblReader);

dsVal = randomPatchExtractionDatastore(voldsVal,pxdsVal,patchSize, ...
    'PatchesPerImage',patchPerImage);
dsVal.MiniBatchSize = miniBatchSize;



dataSource = 'Validation';
dsVal = transform(dsVal,@(patchIn)augmenter3Dcine_MC(patchIn,dataSource,squeeze(coilsets(randi(size(coilsets,1)),:,:,:)) ));



%% network

inputPatchSize = [160 160 24];
numClasses = 4;
[lgraph,outPatchSize] = unet3dLayers(inputPatchSize,numClasses);

outputLayer = dicePixelClassificationLayer('Name','Output');
lgraph = replaceLayer(lgraph,'Segmentation-Layer',outputLayer);

inputLayer = image3dInputLayer(inputPatchSize,'Normalization','none','Name','ImageInputLayer');
lgraph = replaceLayer(lgraph,'ImageInputLayer',inputLayer);


options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.8, ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',500, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'CheckpointPath','/pre_training/training/checkpoints', ...
    'MiniBatchSize',miniBatchSize);


[net,info] = trainNetwork(dsTrain,lgraph,options);

modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');

save(['/pre_training/model/pre_trained_UNET_' modelDateTime  '.mat'],'net','info');


