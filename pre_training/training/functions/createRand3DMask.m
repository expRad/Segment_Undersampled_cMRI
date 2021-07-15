function [mask] = createRand3DMask(sy,sx,sz)
    mask = zeros(sy,sx,sz);
    spokes = [21,34,55,89,144,233,377,610];
    numSpo = spokes(randi(8));
    
    for z=1:sz
        
       mask(:,:,z) = createRadialMask(createRadTrajSim( numSpo, sx).*exp(1i*2*pi*rand(1,1)),sx); 
        
        
    end
    
    
end

