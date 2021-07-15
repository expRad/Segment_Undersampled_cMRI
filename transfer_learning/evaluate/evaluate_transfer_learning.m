clear;
clc;

pathTest = "/transfer_learning/data/dataStore";


%% Test data

volReader = @(x) matRead(x);
volLoc = fullfile(pathTest,'imagesTest');
volds = imageDatastore(volLoc, ...
    'FileExtensions','.mat','ReadFcn',volReader);


lblReader = @(x) segRead(x);
lblLoc = fullfile(pathTest,'labelsTest');
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

dsTest = transform(patchds,@(patchIn)preProcessData(patchIn));

load ('/transfer_learning/models/final_Unet_model.mat');

dicis = zeros(1,5);
SSP = zeros(1,3,4); % specificity, sensitivity, accuracy, precision

EF_LV = zeros(1,2);
EF_RV = zeros(1,2);

ESV_LV = zeros(1,2);
ESV_RV = zeros(1,2);

EDV_LV = zeros(1,2);
EDV_RV = zeros(1,2);

LV_MASS = zeros(1,2);

counter = 1;

while hasdata(dsTest)
    
    tempGroundTruth = read(dsTest);
    groundTruthLabels = tempGroundTruth.inpResponse{1,1};
    vol = tempGroundTruth.inpVol{1,1};

    patchSeg = semanticseg(vol,net);

    %% dice
    
    dicis(counter,1:4) = dice(groundTruthLabels,patchSeg);  % order: background, LV, myo, RV
    dicis(counter,5) = counter;
    
    %% specificity, sensitivity, accuracy, precision

    % LV

    dummy_ref = double(groundTruthLabels);
    dummy_ref(find(dummy_ref~=2)) = 0;
    dummy_ref = imbinarize(dummy_ref);
    
    dummy = double(patchSeg);
    dummy(find(dummy~=2)) = 0;
    dummy = imbinarize(dummy);
    
    addit = dummy_ref + dummy;
    TP = length(find(addit == 2));
    TN = length(find(addit == 0));
    differ = dummy_ref - dummy;
    FP = length(find(differ == -1));
    FN = length(find(differ == 1));

    
    SSP(counter,1,1) = TN/(TN+FP);                  % specificity
    SSP(counter,1,2) = TP/(TP+FN);                  % Sensitivity
    SSP(counter,1,3) = (TP+TN)/(FN+FP+TP+TN);       % Accuracy
    SSP(counter,1,4) = TP/(TP+FP);                  % Precision
    
    
    % myo
    
    dummy_ref = double(groundTruthLabels);
    dummy_ref(find(dummy_ref~=3)) = 0;
    dummy_ref = imbinarize(dummy_ref);
    
    dummy = double(patchSeg);
    dummy(find(dummy~=3)) = 0;
    dummy = imbinarize(dummy);
    
    addit = dummy_ref + dummy;
    TP = length(find(addit == 2));
    TN = length(find(addit == 0));
    differ = dummy_ref - dummy;
    FP = length(find(differ == -1));
    FN = length(find(differ == 1));

    
    SSP(counter,2,1) = TN/(TN+FP);                  % specificity
    SSP(counter,2,2) = TP/(TP+FN);                  % Sensitivity
    SSP(counter,2,3) = (TP+TN)/(FN+FP+TP+TN);       % Accuracy
    SSP(counter,2,4) = TP/(TP+FP);                  % Precision
    
    % RV
    
    dummy_ref = double(groundTruthLabels);
    dummy_ref(find(dummy_ref~=4)) = 0;
    dummy_ref = imbinarize(dummy_ref);
    
    dummy = double(patchSeg);
    dummy(find(dummy~=4)) = 0;
    dummy = imbinarize(dummy);
    
    addit = dummy_ref + dummy;
    TP = length(find(addit == 2));
    TN = length(find(addit == 0));
    differ = dummy_ref - dummy;
    FP = length(find(differ == -1));
    FN = length(find(differ == 1));

    
    SSP(counter,3,1) = TN/(TN+FP);                  % specificity
    SSP(counter,3,2) = TP/(TP+FN);                  % Sensitivity
    SSP(counter,3,3) = (TP+TN)/(FN+FP+TP+TN);       % Accuracy
    SSP(counter,3,4) = TP/(TP+FP);                  % Precision
    
    %% Volumes
    
    % LV
    
    SegNum = 2;
    
    for int = 1:24
        ss(int,1) = size(find(double(groundTruthLabels(:,:,int))==SegNum),1);
        ss(int,2) = size(find(double(patchSeg(:,:,int))==SegNum),1);
    end

    EF_LV(counter,1) = (max(ss(:,1)) - min(ss(:,1)))/max(ss(:,1));
    EF_LV(counter,2) = (max(ss(:,2)) - min(ss(:,2)))/max(ss(:,2));
    
    ESV_LV(counter,1) = min(ss(:,1));
    ESV_LV(counter,2) = min(ss(:,2));
    
    EDV_LV(counter,1) = max(ss(:,1));
    EDV_LV(counter,2) = max(ss(:,2));
    
    % RV
    
    SegNum = 4;
    
    for int = 1:24
        ss(int,1) = size(find(double(groundTruthLabels(:,:,int))==SegNum),1);
        ss(int,2) = size(find(double(patchSeg(:,:,int))==SegNum),1);
    end

    EF_RV(counter,1) = (max(ss(:,1)) - min(ss(:,1)))/max(ss(:,1));
    EF_RV(counter,2) = (max(ss(:,2)) - min(ss(:,2)))/max(ss(:,2));
    
    ESV_RV(counter,1) = min(ss(:,1));
    ESV_RV(counter,2) = min(ss(:,2));
    
    EDV_RV(counter,1) = max(ss(:,1));
    EDV_RV(counter,2) = max(ss(:,2));
    
    
    % MASS
    
    SegNum = 3;
    
    for int = 1:24
        ss(int,1) = size(find(double(groundTruthLabels(:,:,int))==SegNum),1);
        ss(int,2) = size(find(double(patchSeg(:,:,int))==SegNum),1);
    end
    
    LV_MASS(counter,1) = ss(1,1); % (max(ss(:,1)) - min(ss(:,1)))/max(ss(:,1));
    LV_MASS(counter,2) = ss(1,2); %(max(ss(:,2)) - min(ss(:,2)))/max(ss(:,2));
    
    
    %%
    
    counter = counter+1;
    

    
    
end

%% DICE Evaluation

dc = dicis.';

dc = reshape(dc,[5 5 17]);

dice_196 = squeeze(dc(1:4,1,:));
dice_98  = squeeze(dc(1:4,2,:));
dice_49  = squeeze(dc(1:4,3,:));
dice_33  = squeeze(dc(1:4,4,:));
dice_25  = squeeze(dc(1:4,5,:));


LV = zeros(17,5);

LV(:,1) = dice_196(2,:);
LV(:,2) = dice_98(2,:);
LV(:,3) = dice_49(2,:);
LV(:,4) = dice_33(2,:);
LV(:,5) = dice_25(2,:);

[p_LV,tbl_LV,stats_LV] = anova1(LV);


myo = zeros(17,5);

myo(:,1) = dice_196(3,:);
myo(:,2) = dice_98(3,:);
myo(:,3) = dice_49(3,:);
myo(:,4) = dice_33(3,:);
myo(:,5) = dice_25(3,:);

[p_myo,tbl_myo,stats_myo] = anova1(myo);

c = multcompare(stats_myo)


RV = zeros(17,5);

RV(:,1) = dice_196(4,:);
RV(:,2) = dice_98(4,:);
RV(:,3) = dice_49(4,:);
RV(:,4) = dice_33(4,:);
RV(:,5) = dice_25(4,:);

[p_RV,tbl_RV,stats_RV] = anova1(RV);

disp('Mean values LV DSC:')
mean(LV,1)

disp('Mean values Myo DSC:')
mean(myo,1)

disp('Mean values RV DSC:')
mean(RV,1)



% Volumes

EF_LV = EF_LV.';
EF_LV = reshape(EF_LV,[2 5 17]);

EF_RV = EF_RV.';
EF_RV = reshape(EF_RV,[2 5 17]);

ESV_LV = ESV_LV.';
ESV_LV = reshape(ESV_LV,[2 5 17]);

ESV_RV = ESV_RV.';
ESV_RV = reshape(ESV_RV,[2 5 17]);

EDV_LV = EDV_LV.';
EDV_LV = reshape(EDV_LV,[2 5 17]);

EDV_RV = EDV_RV.';
EDV_RV = reshape(EDV_RV,[2 5 17]);

LV_MASS = LV_MASS.';
LV_MASS = reshape(LV_MASS,[2 5 17]);




%% Calculate Relative Errors for Volumes

rel_error_EF_LV = squeeze(abs(EF_LV(1,:,:)-EF_LV(2,:,:))./EF_LV(1,:,:));
disp('Median rel_error_EF_LV: ')
median(rel_error_EF_LV,2)
disp('IQR rel_error_EF_LV: ')
iqr(rel_error_EF_LV,2)


rel_error_EF_RV = squeeze(abs(EF_RV(1,:,:)-EF_RV(2,:,:))./EF_RV(1,:,:));
disp('Median rel_error_EF_RV: ')
median(rel_error_EF_RV,2)
disp('IQR rel_error_EF_RV: ')
iqr(rel_error_EF_RV,2)


rel_error_ESV_LV = squeeze(abs(ESV_LV(1,:,:)-ESV_LV(2,:,:))./ESV_LV(1,:,:));
disp('Median rel_error_ESV_LV: ')
median(rel_error_ESV_LV,2)
disp('IQR rel_error_ESV_LV: ')
iqr(rel_error_ESV_LV,2)


rel_error_ESV_RV = squeeze(abs(ESV_RV(1,:,:)-ESV_RV(2,:,:))./ESV_RV(1,:,:));
disp('Median rel_error_ESV_RV: ')
median(rel_error_ESV_RV,2)
disp('IQR rel_error_ESV_RV: ')
iqr(rel_error_ESV_RV,2)


rel_error_EDV_LV = squeeze(abs(EDV_LV(1,:,:)-EDV_LV(2,:,:))./EDV_LV(1,:,:));
disp('Median rel_error_EDV_LV: ')
median(rel_error_EDV_LV,2)
disp('IQR rel_error_EDV_LV: ')
iqr(rel_error_EDV_LV,2)


rel_error_EDV_RV = squeeze(abs(EDV_RV(1,:,:)-EDV_RV(2,:,:))./EDV_RV(1,:,:));
disp('Median rel_error_EDV_RV: ')
median(rel_error_EDV_RV,2)
disp('IQR rel_error_EDV_RV: ')
iqr(rel_error_EDV_RV,2)


rel_error_LV_MASS = squeeze(abs(LV_MASS(1,:,:)-LV_MASS(2,:,:))./LV_MASS(1,:,:));
disp('Median rel_error_LV_MASS: ')
median(rel_error_LV_MASS,2)
disp('IQR rel_error_LV_MASS: ')
iqr(rel_error_LV_MASS,2)



%% Specificity, Sensitivity, Precision, Accuracy

SSP = permute(SSP,[2 3 1]);
SSP = reshape(SSP,[3 4 5 17]);

disp('Mean LV Specifity: ')
squeeze(mean(SSP(1,1,:,:),4)).'

disp('Mean Myo Specifity: ')
squeeze(mean(SSP(2,1,:,:),4)).'

disp('Mean RV Specifity: ')
squeeze(mean(SSP(3,1,:,:),4)).'


disp('Mean LV Sensitivity: ')
squeeze(mean(SSP(1,2,:,:),4)).'

disp('Mean Myo Sensitivity: ')
squeeze(mean(SSP(2,2,:,:),4)).'

disp('Mean RV Sensitivity: ')
squeeze(mean(SSP(3,2,:,:),4)).'



disp('Mean LV Accuracy: ')
squeeze(mean(SSP(1,3,:,:),4)).'

disp('Mean Myo Accuracy: ')
squeeze(mean(SSP(2,3,:,:),4)).'

disp('Mean RV Accuracy: ')
squeeze(mean(SSP(3,3,:,:),4)).'



disp('Mean LV Precision: ')
squeeze(mean(SSP(1,4,:,:),4)).'

disp('Mean Myo Precision: ')
squeeze(mean(SSP(2,4,:,:),4)).'

disp('Mean RV Precision: ')
squeeze(mean(SSP(3,4,:,:),4)).'


