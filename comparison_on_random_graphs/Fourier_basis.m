%% real fourier basis vectors
function [V] = Fourier_basis(n)
% return an n*n matrix where the columns of V are the Fourier 
% basis vectors in R^n

ell=floor((n-1)/2);

omega=2*pi/n;

V1=sqrt(2)*cos(omega*[0:1:n-1]'*[1:1:ell]);
V2=sqrt(2)*sin(omega*[0:1:n-1]'*[1:1:ell]);

V=[ones(n,1),V1,V2];
if mod(n,2) ==0
    V=[V,cos(pi*[0:1:n-1]')];
end

V=V/sqrt(n);


