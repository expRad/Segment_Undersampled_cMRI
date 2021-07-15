function sigfilled = zerofill(sig, newsize)

sigsize = size(sig);
dim = ndims(sig);
dim_n = length(newsize);

pads = zeros(1,dim);
shifts = zeros(1,dim);

for q = 0 : dim_n-1
    pads(dim-q) = newsize(dim_n-q)-sigsize(dim-q);
    
    shifts(dim-q) = ceil( (newsize(dim_n-q)-sigsize(dim-q))/2 );
end

sigfilled = padarray(sig,pads,0,'pre');

sigfilled = FouShift(sigfilled,shifts);
