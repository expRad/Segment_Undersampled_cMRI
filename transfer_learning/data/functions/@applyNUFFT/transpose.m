function out = transpose(a)
a.adjoint = xor(a.adjoint,1);
out = a;