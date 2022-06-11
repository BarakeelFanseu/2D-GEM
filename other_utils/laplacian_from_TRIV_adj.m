function [L, evecs, evals] = laplacian_from_TRIV_adj(adj, numEigs)
% 
% if nargin > 1 && numEigs > 0
%     adj = adj + eye(size(adj, 1)); % self loops to recondision L for good computation
% end

deg = sparse(adj) * ones(size(adj, 1),1);

% disp('symmetric normalized laplacian used')
deg = 1./sqrt(deg);
deg(deg==Inf) = 0;
deg(deg==-Inf) = 0;
deg = sparse(diag(deg));
L = sparse(eye(size(adj, 1)) - (deg * adj * deg));
if nargin > 1 && numEigs > 0
%     [evecs, evals] = eigs(L, numEigs, 1e-6);
    [evecs, evals] = eigs(L, numEigs, 1e-6);
    evals = diag(evals);

else
    fprintf('NO laplacian_Eigen-functions used \n');
    evecs = 0;
    evals = 0;
end

[evals, order] = sort(abs(evals),'ascend');
evals = evals.^10;
evecs = evecs(:,order);



