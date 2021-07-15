function y=iFouShift(x,shifts)
numDims = ndims(x);				
idx = cell(1, numDims);												
for k = 1:numDims
    m = size(x, k);
    p = floor(shifts(k));      
	if p < 0
		p=m+p;
    end
    idx{k} = [p+1:m 1:p];
end
y = x(idx{:});
