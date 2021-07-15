function [mask] = create3DMaskSpokes(sy,sx,sz,spokes)

    mask = zeros(sy,sx,sz);
    
    for z=1:sz
        
       mask(:,:,z) = createRadialMask(createRadTrajSim( spokes, sx).*exp(1i*2*pi*rand(1,1)),sx); 
        
        
    end
    
    
end

