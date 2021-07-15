function img=ifft2dAI(sig)
dim=size(sig);
shift=zeros(1,size(dim,2));
shift(1,2)=dim(2)/2;
shift(1,1)=dim(1)/2;
img=iFouShift(ifft(ifft(iFouShift(sig,shift),[],2),[],1),shift);