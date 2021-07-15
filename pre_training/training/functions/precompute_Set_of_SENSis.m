function [setofsensis] = precompute_Set_of_SENSis(coils, numberS)

     setofsensis = zeros(numberS,256,256,coils);
     
     angls = (2*pi/coils):(2*pi/coils):2*pi;
     
     disp('Simulating Coil Sensitivities...');
     
     for setti = 1:numberS
         
         anglsi = angls+(pi/coils)*randn(1,1);
         
         radic = 0.4 + ((rand(1,1)-0.5)*0.1);
         
         distc = 1.5 + ((rand(1,1)-0.5)*0.2);
         
         setofsensis(setti,:,:,:) = GenerateSensitivityMap([0.512,0.512],[0.002,0.002],anglsi,radic,distc);

     end
     
     disp('Done.');
     

end

