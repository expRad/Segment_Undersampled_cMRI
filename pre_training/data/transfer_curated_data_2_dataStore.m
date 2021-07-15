clear;
clc;

tbl = readtable('15percent_images.txt','ReadVariableNames',false,'Delimiter','\n');
tbl.Properties.VariableNames = {'filenames'};

disp('This takes a few hours...');

%%

f = waitbar(0,'Parsing key parameters.','Name','Please wait ...');

for bil = 1:size(tbl,1)
    
    tbl{bil,2} = str2double(extractBefore(tbl.filenames{bil},'-image'));
    tbl{bil,3} = str2double(extractBetween(tbl.filenames{bil},'slice','.png'));
    tbl{bil,4} = str2double(extractBetween(tbl.filenames{bil},'frame','-slice'));
    
    waitbar(bil/size(tbl,1),f,'Parsing key parameters.');
    
end

close(f)

%%

patients  = tbl.Var2;

[a,b] = hist(patients,1:1:max(patients));

patIDs = b(find(a));

volCounter = 1;

f = waitbar(0,'Constructing Cine Series.','Name','Please wait ...');

for pat = 1:size(patIDs,2)   
    
    slices = max(tbl.Var3(find(tbl.Var2==patIDs(pat))));
       
    for slice = 0:slices
        
       imVol = zeros(256,256,30);
        
       for frame = 0:29
           
           idx =  intersect(intersect(find(tbl.Var2==patIDs(pat)), find(tbl.Var3==slice)), find(tbl.Var4==frame));
           
           %disp(tbl.filenames{idx})
           
           bld = imread(['/pre_training/data/pngs/' tbl.filenames{idx}]);
           
           [y,x,~] = size(bld);
           bld = zerofill(bld,[max([y,x]) max([y,x]) 3]);

           imVol(:,:,frame+1) = single(imresize(bld(:,:,1),[256 256]));
       end
       
       imVol = normalize_images(imVol);
       save(['/pre_training/data/dataStoreKaggle/images/imVol' num2str(volCounter)]  ,'imVol')
       
       volCounter = volCounter+1;
        
    end
    
    waitbar(pat/size(patIDs,2),f,'Constructing Cine Series.');
    
end

close(f)

%% Splitting data to train, validate and test
disp('Splitting data.');

allIms = dir('/pre_training/data/dataStoreKaggle/images/*.mat');

for immi=1:size(allIms,1)
    
   tt = extractBetween(allIms(immi).name,'imVol','.mat'); 
    
   if( str2num(tt{1}) >4300 && str2num(tt{1}) < 4451)
       
       movefile(fullfile(allIms(immi).folder,allIms(immi).name) , fullfile(strrep(allIms(immi).folder,'images','imagesVal'),allIms(immi).name));
       
   elseif (str2num(tt{1}) >4450)
       
       movefile(fullfile(allIms(immi).folder,allIms(immi).name) , fullfile(strrep(allIms(immi).folder,'images','imagesTest'),allIms(immi).name));
        
   end 
    
    
end



disp('Finished.');

