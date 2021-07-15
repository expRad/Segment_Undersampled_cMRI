function [out] = normalize_images(in)
% remove the mean and divide by the std

chn_Mean = mean(in,[1 2 3]);
chn_Std = std(in,0,[1 2 3]);
out = (in - chn_Mean)./chn_Std;

rangeMin = -5;
rangeMax = 5;

out(out > rangeMax) = rangeMax;
out(out < rangeMin) = rangeMin;

out = (out - rangeMin) / (rangeMax - rangeMin);
end

