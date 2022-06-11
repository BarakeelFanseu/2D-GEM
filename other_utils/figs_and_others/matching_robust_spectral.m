% Improved implementation of our new robust spectral method 
% A and B are the matrices to be matched 
% eta is the tuning parameter comparable to noise level sigma
% Return permutation matrix P so that P*A*P' is matched to B 

function [P_sp] = matching_robust_spectral(shapeA, shapeB, eta)
    disp('=======Using GRAMPA=========')
    n = size(shapeA.adj, 1);
    m = size(shapeB.adj,1);
    %[U, Lambda] = eig(shapeA.adj);
    %[V, Mu] = eig(shapeB.adj);
    
    %% shapeA
    U = shapeA.adj_evecs;
    Lambda = shapeA.adj_evals;
    
    %% shapeB
    V = shapeB.adj_evecs;
    Mu = shapeB.adj_evals;
    
    %% spectral filtering
    lambda = diag(Lambda);
    mu = diag(Mu);
    
    Coeff = 1 ./ ((lambda - mu').^2 + eta^2);    
    Coeff = Coeff .* (U' * ones(n,m) * V);
    X = U * Coeff * V';
    
    P_sp = greedy_match(X);  
    
    %% Rounding by linear assignment - better but slower 
%    M = matchpairs(X', -99999, 'max');
%    P = full(sparse(M(:, 1), M(:, 2), 1, n, n));
%    C=X-min(min(min(X),0));
%    C=C*10^6;
%    C=round(C);
%    [assignments,P]=sparseAssignmentProblemAuctionAlgorithm(C');
    %% Greedy matching - faster but worse 
%    P = full(greedy_match_better(X',A,DA_model,DB_model));

    %% Greedy rounding
%     [~, ind_max] = max(X);
%     P = full(sparse(1:n, ind_max, 1, n, n));