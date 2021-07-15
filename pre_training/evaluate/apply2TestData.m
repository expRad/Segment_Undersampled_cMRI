function [dicis,EF_list,SSP] = apply2TestData(net,projs,path2TestData, meta_data, eval_list)

volReader = @(x) matRead(x);
classNames = ["background","LV","myo","RV"];
pixelLabelID = [0 1 2 3];
patchSize = [256 256 24];
patchPerImage = 1;
miniBatchSize = 1;

volLocTest = fullfile(path2TestData,'imagesTest');
voldsTest = imageDatastore(volLocTest, ...
    'FileExtensions','.mat','ReadFcn',volReader);

lblReader = @(x) segRead(x);
lblLocTest = fullfile(path2TestData,'labelsTest');
pxdsTest = pixelLabelDatastore(lblLocTest,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',lblReader);

dsTest = randomPatchExtractionDatastore(voldsTest,pxdsTest,patchSize, ...
    'PatchesPerImage',patchPerImage);
dsTest.MiniBatchSize = miniBatchSize;

coilsets = precompute_Set_of_SENSis(12,5);

dataSource = 'Test';

dsTest = transform(dsTest,@(patchIn)pre_process_data(patchIn,dataSource,projs, squeeze(coilsets(randi(size(coilsets,1)),:,:,:)) ));
id = 1;

dicis = zeros(100,6);

SSP = zeros(1,3,4); % specificity, sensitivity, accuracy, precision

volumes = zeros(100,2,24);

counter = 1;

disp('Applying neural network to test data...');

while hasdata(dsTest)

    tempGroundTruth = read(dsTest);
    groundTruthLabels = tempGroundTruth.inpResponse{1,1};
    vol = tempGroundTruth.inpVol{1,1};

    patchSeg = semanticseg(vol,net);
    
    dicis(counter,1:4) = dice(groundTruthLabels,patchSeg);
    
    dicis(counter,5) = meta_data(4450+counter,4);
    dicis(counter,6) = meta_data(4450+counter,2);
    
    for frame = 1:24
        volumes(counter,1,frame) = size(find(double(groundTruthLabels(:,:,frame))==2),1);
        volumes(counter,2,frame) = size(find(double(patchSeg(:,:,frame))==2),1);
    end
    
    
    
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
    SSP(counter,1,2) = TP/(TP+FN);                  % sensitivity
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
    SSP(counter,2,2) = TP/(TP+FN);                  % sensitivity
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
    SSP(counter,3,2) = TP/(TP+FN);                  % sensitivity
    SSP(counter,3,3) = (TP+TN)/(FN+FP+TP+TN);       % Accuracy
    SSP(counter,3,4) = TP/(TP+FP);                  % Precision
    
    counter = counter+1;
    
end

disp('Done.');

disp('Determining volumes...');

%% analyze volumes

patslist = unique(dicis(:,6));

EF_list = zeros(size(patslist,1),2);

for patti = 2:size(patslist,1)
    
    a = find(dicis(:,6) == patslist(patti));
    b = find(eval_list == 1);
    
    c = intersect(a,b);
    
    
    vol1 = 0;
    vol2 = 0;
    for tt = c(1):c(end)

        [~,po] = min(volumes(tt,1,:));
        [~,po2] = min(volumes(tt,2,:));

        vol1 = vol1 + volumes(tt,1,po);
        vol2 = vol2 + volumes(tt,2,po2);
    end
    vol3 = 0;
    vol4 = 0;
    for tt = c(1):c(end)
        [~,po] = max(volumes(tt,1,:));
        [~,po2] = max(volumes(tt,2,:));

        vol3 = vol3 + volumes(tt,1,po);
        vol4 = vol4 + volumes(tt,2,po2);
    end


    EF_GT = (vol3-vol1)/vol3;
    EF_UNET = (vol4-vol2)/vol4;
    
    EF_list(patti,1) = EF_GT;
    EF_list(patti,2) = EF_UNET;
    
    
end

disp('Done.');

end