# Semantic segmentation of radially undersampled cardiac MRI

This repository contains instructions and source code to reproduce the results presented in:

> A data-driven semantic segmentation model for direct cardiac functional analysis based on undersampled radial MR cine series.  
> Wech T, Ankenbrand MJ, Bley TA, Heidenreich JF.  
> Magnetic Resonance in Medicine. 2021. *Early view*. [DOI](https://doi.org/10.1002/mrm.29017)

Please cite this work if you use the content of this repository in your project.

The vast majority of codes is written in MATLAB (R2019b, Deep Learning Toolbox). Solely the pre-processing of kaggle images is provided as python software.

## Pre-training using data from the Second Annual Data Science Bowl

The segmentation model was first pre-trained using data from [Kaggle's Second Annual Data Science Bowl](https://www.kaggle.com/c/second-annual-data-science-bowl) (Transforming How We Diagnose Heart Disease) presented by Booz Allen Hamilton. 

1.  As a first step you need to download the challenge data located here: https://www.kaggle.com/c/second-annual-data-science-bowl/data. Please note citation remark for data usage at the bottom of the page. Extract the dataset at ***/pre_training/data/kaggle_download***.

2.  Data curation as described in

    >Deep learning-based cardiac cine segmentation: Transfer learning application to 7T ultrahigh-field MRI  
    >Ankenbrand MJ, Lohr D, Schlötelburg W, Reiter T, Wech T, Schreiber LM  
    >Magnetic Resonance in Medicine. 2021. *Early view*. [DOI](https://doi.org/10.1002/mrm.28822)

    was performed to obtain pairs of images and labels with highest fidelity. This requires several sub-steps as detailed below:
    
    1.  The python script `/pre_training/data/dicom2nifti.py` first converts dicoms from the kaggle datasets to nifti format. Make sure you have the following libraries 
        installed:
        - pydicom  
        - nibabel  
        - numpy  
        - tqdm


        Execute the following command:  
            

        ```{bash}
        mkdir python_conversion
        python dicom2nifti.py | tee python_conversion/conversion.log
        ```
      

        Now there is one folder for each patient from the kaggle dataset in the python_conversion folder containing a single nifti each.
        The files **conversion.log** and **used_dicoms.log** contain information about which images were used and the reasons for excluding files.
       
    2.  We then convert images using [`med2image`](https://github.com/FNNDSC/med2image)  
    
        ```  
        # Convert images
        for i in python_conversion/*/sa.nii.gz
        do
            med2image -i $i -d pngs -o $(basename $(dirname $i))-image.png -t png
        done
        ```    
    3.  Images with good segmentation performance (`/pre_training/data/15percent_images.txt`) will then be selected and stored in the final data store at ***/pre_training/data/dataStoreKaggle*** using the MATLAB script `/pre_training/data/transfer_curated_data_2_dataStore.m`. Separate folders for training, validation and test are organized (**images**, **imagesVal** and **imagesTest**).


3.  We determined segmentation labels using the model provided by Wenjia Bai. As described in

    >Automated cardiovascular magnetic resonance image analysis with fully convolutional networks  
    >Bai W, Sinclair M, Tarroni G, Oktay O, Rajchl M, Vaillant G, Lee AM, Aung N, Lukaschuk E, Sanghvi MM, Zemrak F, Fung K, Paiva JM, Carapella V, Kim YJ, Suzuki H, Kainz B,      Matthews PM, Petersen SE, Piechnik SK, Neubauer S, Glocker B, Rueckert D  
    >Journal of Cardiovascular Magnetic Resonance. 2018; 20:65  

    the authors used a large-scale annotated dataset from [UK Biobank](https://www.ukbiobank.ac.uk/) to train a FCN which achieved a performance comparable with human experts.       This model can be downloaded as part of the toolbox provided here:

    https://github.com/baiwenjia/ukbb_cardiac

    We applied this model to the (curated) kaggle data and uploaded the resulting labels to the folders **labels**, **labelsVal** and **labelsTest** in ***/pre_training/data/dataStoreKaggle***. Please also cite [Wenjia's work](https://github.com/baiwenjia/ukbb_cardiac#references) if you use the derived labels in your project.
    

4.  Run the script `/pre_training/training/pre_train_network.m` to train the Unet architecture with the dataset prepared by the aforementioned steps. Undersampled datasets are simulated on-the-fly during training by means of nested functions (see code for details). To enable the required simulation of coil sensitivities, you need to download Matthieu Guerquin-Kern's [MATLAB toolbox](http://bigwww.epfl.ch/algorithms/mri-reconstruction/code_v1-0.zip) and add `GenerateSensitivityMap.m` to the MATLAB path. The model trained for the publication (using `/pre_training/training/pre_train_network.m`) is stored in ***/pre_training/model/pre_trained_UNET.mat***.

5. For evaluation of the model use `/pre_training/evaluate/eval_pre_trained_Unet.m`, which results in the scores presented in the paper.

## Transfer learning using radially acquired cine data from Harvard Medical School

In the second part of the paper we describe how to transfer the pre-trained model towards robust application for realistic radially undersampled cine data.
For this purpose we used radial MR data provided by Reza Nezafat's research group in conjunction with their paper:

> Multi-domain convolutional neural network (MD-CNN) for radial reconstruction of dynamic cardiac MRI. 
> El-Rewaidy H, Fahmy AS, Pashakhanloo F, Cai X, Kucukseymen S, Csecs I, Neisius U, Haji-Valizadeh H, Menze B, Nezafat R.  
> Magnetic Resonance in Medicine. 2021; 85:1195–1208.  

The following steps need to be performed, to reproduce the results of the optimized segmentation model:

1.  Download all *.mat files from https://doi.org/10.7910/DVN/CI3WB6 and save this set in `/transfer_learning/data/raw_data`. Please take note of the terms for data usage as detailed there. See https://doi.org/10.1002/mrm.28485 for more information on the dataset. 
    
2.  For reconstructing non-Cartesian MR data, Jeffrey Fessler's non-uniform FFT MATLAB implementation is used. Download the Michigan Image Reconstruction Toolbox using this [link](https://web.eecs.umich.edu/~fessler/irt/irt/) and add content to your path by executing the setup.m script within the root folder.
  
3.  To reconsruct the radial MR cine raw data for several different factors of undersampling, execute the script `/transfer_learning_/data/reconstruct_data.m`. A few cases are excluded by the script due to limited quality of the labels (as evaluated by an expert radiologist), which were obtained from Bai's semantic segmentation model (see step 5.). Reconstructed image series are stored in ***/transfer_learning_/data/recons***.

4.  Image series are formatted and assigned to train, validate and test datastores (***/transfer_learning_/data/dataStore***) by means of `/transfer_learning_/data/preprocess_and_separate_data.m`.

5.  According segmentation labels were again determined using Wenjia Bai's segmentation network which was trained using [UK Biobank](https://www.ukbiobank.ac.uk/) Data (same procedure as described in 3. of the pre training section). The according toolbox can be found here: https://github.com/baiwenjia/ukbb_cardiac. The model was applied to the fully sampled versions of the cine series, and labels were already stored within the folders **labels**, **labelsVal** and **labelsTest** in ***/transfer_learning_/data/dataStore***.

6.  Execute the script `/transfer_learning_/training/transfer_learning.m` to perform the transfer learning step using the model trained by the kaggle data for initial weights and the genuine radial cine data from Harvard Medical School, prepared as described above.

7.  Use the script `/transfer_learning_/evaluate/evaluate_transfer_learning.m` to determine dice scores etc. as presented in our manuscript.




