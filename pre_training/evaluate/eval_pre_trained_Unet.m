clear;
clc;

load /pre_training/model/pre_trained_UNET.mat
TestDataPath = "/pre_training/data/dataStoreKaggle/";

load /pre_training/evaluate/meta_data.mat
load /pre_training/evaluate/slicelist.mat

reps = 5;  % repeated evaluation for better estimation as test data run through random data augmentation

%% Evaluate performance on kaggle test data with Unet trained by kaggle data only

diceM = zeros(reps,6,500,6);
EF_M = zeros(reps,6,49,2);
SSP_M = zeros(reps,6,500,3,4);

pcount = 1;

for projs = [21, 34, 55, 89, 144, 377]
    
    for rep = 1:reps
       
        tic
        
        disp(['Evaluating repetition ' num2str(rep) ' of #projection = ' num2str(projs) '.']);
    
        [dicis,EF_list,SSP] = apply2TestData(net,projs, TestDataPath, meta_data, eval_list);
        
        diceM(rep,pcount,:,:) = dicis;
        EF_M(rep,pcount,:,:) = EF_list;
        SSP_M(rep,pcount,:,:,:) = SSP;
        
        disp(['Iteration took ' num2str(toc/60) ' minutes.'])
    
    end
    
    pcount = pcount + 1;
    
end


save('stats_KAGGLE_model','diceM','EF_M','SSP_M');


clear;
clc;

load C:\Users\wech_t\Documents\GitHub\Seg_Undersampled_cMRI\transfer_learning\models\final_Unet_model.mat
TestDataPath = "Z:\twech\DeepLearning\MA\createDataStores\perc15_2_MULTI_LABEL";

load meta_data.mat
load slicelist.mat

reps = 5;  % repeated evaluation for better estimation as test data run through random data augmentation

%% Evaluate performance on kaggle test data with Unet trained by kaggle data only

diceM = zeros(reps,6,500,6);
EF_M = zeros(reps,6,49,2);
SSP_M = zeros(reps,6,500,3,4);

pcount = 1;

for projs = [21, 34, 55, 89, 144, 377]
    
    for rep = 1:reps
       
        tic
        
        disp(['Evaluating repetition ' num2str(rep) ' of #projection = ' num2str(projs) '.']);
    
        [dicis,EF_list,SSP] = apply2TestData(net,projs, TestDataPath,meta_data,eval_list);
        
        diceM(rep,pcount,:,:) = dicis;
        EF_M(rep,pcount,:,:) = EF_list;
        SSP_M(rep,pcount,:,:,:) = SSP;
        
        disp(['Iteration took ' num2str(toc/60) ' minutes.'])
    
    end
    
    pcount = pcount + 1;
    
end


save('stats_HARVARD_model','diceM','EF_M','SSP_M');



return



load stats_KAGGLE_model.mat
load slicelist.mat

%% DICE


evlist = find(eval_list==1);

dice377 = squeeze(mean(diceM(:,6,:,:),1));
dice377  = nanmean(dice377(evlist,1:4))


dice144 = squeeze(mean(diceM(:,5,:,:),1));
dice144  = nanmean(dice144(evlist,1:4))


dice89 = squeeze(mean(diceM(:,4,:,:),1));
dice89  = nanmean(dice89(evlist,1:4))


dice55 = squeeze(mean(diceM(:,3,:,:),1));
dice55  = nanmean(dice55(evlist,1:4))


dice34 = squeeze(mean(diceM(:,2,:,:),1));
dice34  = nanmean(dice34(evlist,1:4))


dice21 = squeeze(mean(diceM(:,1,:,:),1));
dice21  = nanmean(dice21(evlist,1:4))


%% Sensitivity,....

evlist = find(eval_list==1);

SSP_M = squeeze(nanmean(mean(SSP_M(:,:,evlist,:,:),1),3));

disp(['Mean specificity LV: ']);

fliplr(SSP_M(:,1,1).')

disp(['Mean sensitivity LV: ']);

fliplr(SSP_M(:,1,2).')

disp(['Mean accuracy LV: ']);

fliplr(SSP_M(:,1,3).')

disp(['Mean precision LV: ']);

fliplr(SSP_M(:,1,4).')


disp(['Mean specificity Myo: ']);

fliplr(SSP_M(:,2,1).')

disp(['Mean sensitivity Myo: ']);

fliplr(SSP_M(:,2,2).')

disp(['Mean accuracy Myo: ']);

fliplr(SSP_M(:,2,3).')

disp(['Mean precision Myo: ']);

fliplr(SSP_M(:,2,4).')


disp(['Mean specificity RV: ']);

fliplr(SSP_M(:,3,1).')

disp(['Mean sensitivity RV: ']);

fliplr(SSP_M(:,3,2).')

disp(['Mean accuracy RV: ']);

fliplr(SSP_M(:,3,3).')

disp(['Mean precision RV: ']);

fliplr(SSP_M(:,3,4).')

%% Apply Harvard Model to Kaggle data

load stats_HARVARD_model.mat
load slicelist.mat

%% DICE


evlist = find(eval_list==1);

dice377 = squeeze(mean(diceM(:,6,:,:),1));
dice377  = nanmean(dice377(evlist,1:4))


dice144 = squeeze(mean(diceM(:,5,:,:),1));
dice144  = nanmean(dice144(evlist,1:4))


dice89 = squeeze(mean(diceM(:,4,:,:),1));
dice89  = nanmean(dice89(evlist,1:4))


dice55 = squeeze(mean(diceM(:,3,:,:),1));
dice55  = nanmean(dice55(evlist,1:4))


dice34 = squeeze(mean(diceM(:,2,:,:),1));
dice34  = nanmean(dice34(evlist,1:4))


dice21 = squeeze(mean(diceM(:,1,:,:),1));
dice21  = nanmean(dice21(evlist,1:4))

