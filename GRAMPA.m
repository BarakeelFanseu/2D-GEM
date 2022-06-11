function [pertF] = GRAMPA(surf1, surf2, options, eta)

%% initialize 2-hop
n1 = surf1.n;
n2 = surf2.n;
EYE1=sparse(1:n1,1:n1,1,n1,n1);
EYE2=sparse(1:n2,1:n2,1,n2,n2);
W21=sparse((double((surf1.adj*surf1.adj)>0)-surf1.adj-EYE1)>0);
W22=sparse((double((surf2.adj*surf2.adj)>0)-surf2.adj-EYE2)>0);

%% initialize filter
[U, Lambda] = eigs(double(surf2.adj), n2);
[V, Mu] = eigs(double(surf1.adj), n1);
lambda = diag(Lambda);
mu = diag(Mu);
Coeff = 1 ./ ((lambda - mu').^2 + eta);
% imagesc(Coeff); axis square;%%

Coeff = Coeff .* (U' * ones(surf2.n, surf1.n) * V);
pertF = U * Coeff * V';
pertF = greedy_match(pertF); 
    
%% 2 hop improve
%     a = zeros(options.maxIter, 1);%%
for iter=1:1:options.maxIter
    pertF = greedy_match(W22*pertF*W21);
%         a(iter) = trace(pertF'*surf2.adj*pertF*surf1.adj);%%
%         disp(a(iter))%%
end