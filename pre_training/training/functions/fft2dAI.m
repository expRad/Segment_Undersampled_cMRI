function sig=fft2dAI(img);
dim=size(img);
shift=zeros(1,size(dim,2));
shift(1,2)=dim(2)/2;
shift(1,1)=dim(1)/2;
sig=FouShift(fft(fft(FouShift(img,shift),[],2),[],1),shift);