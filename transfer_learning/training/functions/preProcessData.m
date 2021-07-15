function patchOut = preProcessData(patchIn)

inpVol = cell(size(patchIn,1),1);
inpResponse = cell(size(patchIn,1),1);


for id=1:size(patchIn,1) 

    tmpImg =  patchIn.InputImage{id};
    tmpResp = patchIn.ResponsePixelLabelImage{id};

    out =  tmpImg;
    respOut = tmpResp;

    respFinal = respOut(:,:,1:24);
    
    out = out/1.8; % scaling
    
    wind = 49:208;
    
    respFinal = respFinal(wind,wind,:);
    out = out(wind,wind,:);
    
    out = normalize_images(out);
    
    inpVol{id,1}= out;
    inpResponse{id,1}=respFinal;
end

patchOut = table(inpVol,inpResponse);