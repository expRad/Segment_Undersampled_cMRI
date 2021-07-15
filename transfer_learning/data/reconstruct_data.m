clear;
clc;
 
rawdatapath = '/transfer_learning/data/raw_data';  % path to data as downloaded from https://doi.org/10.7910/DVN/CI3WB6

reconpath = '/transfer_learning/data/recons/';

data_list = dir([rawdatapath '*.mat']);  

load ('results_rating.mat');

for dataset = 1:size(data_list,1)
    
   index = find(results_rating(:,1)==dataset);

   if(results_rating(index,2))
    
       load([rawdatapath  data_list(dataset).name]);

       disp(data_list(dataset).name)

       data = squeeze(data);

       data = data(:,:,:,:,1) + 1i * data(:,:,:,:,2);

       data = permute(data,[4 1 3 2]);

       traj = createRadTraj(size(data,3),416);

       cine = zeros(25,208,208);
       
       bl = -1:0.00481:1;
       bl = abs(bl);
       w = zeros(size(traj));
       for n=1:size(traj,1)
         w(n,:) = bl;
       end

       
       
       for R=[1 2 4 6 8]
           
           disp(['Reconstructing for acceleration factor: ' num2str(R)'.'])
           
           for phase = 1:25
                      
               
               acqu = squeeze(data(:,phase,(mod(phase-1,R)+1):R:end,209:624));
               tr_temp = traj((mod(phase-1,R)+1):R:end,:);
               
               N = [416,416];
               [xx,yy] = meshgrid(linspace(-1,1,N(1)));
               ph = double(sqrt(xx.^2 + yy.^2)<1);

               FT = applyNUFFT(tr_temp(:),w(1:size(tr_temp,1),:),ph, 0,N, 2);
               
               temp = zeros(size(data,1),416,416);
               
               for coil = 1:size(data,1)
                   temp(coil,:,:) = FT'*squeeze(acqu(coil,:,:));
               end
              
               cine(phase,:,:) = permute(rssq(temp(:,105:312,105:312),1),[1 3 2]);
               
           end
           
           datname = data_list(dataset).name;
           save([reconpath datname(1:4) '_R' num2str(R)  ],'cine');
           
       end
       

       
   end
    
end