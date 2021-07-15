function outi = times(a,bb)
for n=1:size(bb,3)
b = bb(:,:,n);
if a.adjoint
	b = b(:).*a.w(:);
	out = nufft_adj(b, a.st)/sqrt(prod(a.imSize));
	out = reshape(out, a.imSize(1), a.imSize(2));
	out = fft2c(out);
else
	b = reshape(b,a.imSize(1),a.imSize(2));
	b = ifft2c(b);
	out = nufft(b, a.st)/sqrt(prod(a.imSize));
	out = reshape(out,a.dataSize(1),a.dataSize(2));
end
outi(:,:,n) = out;
end

