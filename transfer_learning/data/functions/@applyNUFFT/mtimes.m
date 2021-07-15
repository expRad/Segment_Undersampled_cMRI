function outi = mtimes(a,bb)
for n=1:size(bb,3)
b = bb(:,:,n);

if a.adjoint
	b = b(:).*a.weight(:);
	out = nufft_adj(b, a.st)/sqrt(prod(a.matS));
	out = reshape(out, a.matS(1), a.matS(2));
	out = out.*conj(a.phase);
	if a.mode==1
		out = real(out);
	end

else
	b = reshape(b,a.matS(1),a.matS(2));
	if a.mode==1
        b = real(b);
	end
	b = b.*a.phase;
	out = nufft(b, a.st)/sqrt(prod(a.matS));
	out = reshape(out,size(a.weight,1),size(a.weight,2));

end
outi(:,:,n) = out;
end

