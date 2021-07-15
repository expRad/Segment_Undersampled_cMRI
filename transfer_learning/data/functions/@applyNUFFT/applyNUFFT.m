function  out = applyNUFFT(k,weight,phase,shift,matS, mode)

    klist = [real(k(:)), imag(k(:))]*2*pi;
    Nd = matS;
    Jd = [6,6];
    Kd = [Nd*2];
    n_shift = Nd/2 + shift;
    out.st = nufft_init(klist, Nd, Jd, Kd, n_shift);

    out.phase = phase;
    out.adjoint = 0;
    out.matS = matS;
    out.dataSize = size(k);
    out.weight = weight;
    out.mode = mode;
    out = class(out,'applyNUFFT');

