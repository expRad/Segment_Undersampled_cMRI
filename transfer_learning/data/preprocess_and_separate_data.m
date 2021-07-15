clear;
clc;

recos = dir('/transfer_learning/data/recons/*.mat'); % data has to be reconstructed using reconstruct_data.m before

image_path = '/transfer_learning/data/dataStore/';

flip_lr_list = [17,18,19,24,35,38,42,43,45,47,50,51,54,61,66,69,73,81,87,89,91,94,97];

f = waitbar(0,'Processing and assigning reconstructed data.','Name','Please wait ...');

for re = 1:size(recos,1)
     
    
   numb = extractBetween(recos(re).name,'P','_R');
   
   if contains(numb,'_') 
       numb = erase(numb,'_'); 
   end
   
   numb = str2double(numb);
    
   load(fullfile(recos(re).folder,recos(re).name));
   
   imVol = zeros(25,256,256);
   
   for in = 1:25
       if(find(numb == flip_lr_list))
           
           imVol(in,:,:) = fliplr(flipud(imresize(squeeze(cine(in,:,:)),[256 256])));
           
       else
           
           imVol(in,:,:) = flipud(imresize(squeeze(cine(in,:,:)),[256 256]));
            
       end
       
   end
   
   imVol = permute(normalize_images(imVol),[2 3 1]);
   
   
   
   if numb <=78  % training data
           
      save([image_path 'images\' recos(re).name],'imVol');
      
   elseif numb >= 79 && numb <= 84  % validation data
       
      save([image_path 'imagesVal\' recos(re).name],'imVol');
      
   elseif numb >= 85 && numb <= 108 % test data
       
      save([image_path 'imagesTest\' recos(re).name],'imVol');
      
   else
       
       disp('Error')
       
       
   end
   
   waitbar(re/size(recos,1),f,'Processing and assigning reconstructed data.');
    
end
    
close(f) 