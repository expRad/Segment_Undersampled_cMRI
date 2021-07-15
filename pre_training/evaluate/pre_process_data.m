function patchOut = pre_process_data(patchIn,flag,spokes,coilset)

isValidationData = strcmp(flag,'validation');

inpVol = cell(size(patchIn,1),1);
inpResponse = cell(size(patchIn,1),1);

fliprot = @(x) rot90(fliplr(x));
augType = {@rot90,@fliplr,@flipud,fliprot};
for id=1:size(patchIn,1) 
    rndIdx = randi(8,1);
    tmpImg =  patchIn.InputImage{id};
    tmpResp = patchIn.ResponsePixelLabelImage{id};
    if rndIdx > 4 || isValidationData
        out =  tmpImg;
        respOut = tmpResp;
    else
        out =  augType{rndIdx}(tmpImg);
        respOut = augType{rndIdx}(tmpResp);
    end
    
    
    respFinal = respOut;
    
    bild = repmat(out,[1 1 1 size(coilset,3)]).*permute(repmat(coilset,[1 1 1 24]),[1 2 4 3]);
    
    mask = repmat(create3DMaskSpokes(256,256,24,spokes),[1 1 1 size(coilset,3)]);
    out = mask.*fft2dAI(bild);
    
    out=abs(squeeze(rssq(ifft2dAI(out),4))); %respOut(49:end-48,49:end-48,4:end-3,:);
    
    out = out/1.8; % scale below 1 
    

    wind = 49:208;
    
    respFinal = respFinal(wind,wind,:);
    out = out(wind,wind,:);
    
    
    inpVol{id,1}= out;
    inpResponse{id,1}=respFinal;
end
patchOut = table(inpVol,inpResponse);