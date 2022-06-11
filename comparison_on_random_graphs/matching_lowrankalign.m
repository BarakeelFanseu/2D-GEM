% LowRankAlign
% A and B are the matrices to be matched 
% Return permutation matrix P so that P*A*P' is matched to B 

function [P_best] = matching_lowrankalign(A, B, k)
    n = size(A, 1);
    [U, Lambda] = eigs(A, k, 'largestreal');
    [V, Mu] = eigs(B, k, 'largestreal');
    P_best = eye(n);
    
    SS = dec2bin(0:2^k-1) - '0';
    for j = 1:2^k
        s = SS(j, :) * 2 - 1;
        X = zeros(n);
        for i = 1:k
            X = s(i) * Lambda(i, i) * Mu(i, i) * U(:, i) * V(:, i)';
        end
        
        M = matchpairs(X', -99999, 'max');
        P = full(sparse(M(:, 1), M(:, 2), 1, n, n));
        if sum(dot(P*A*P', B)) > sum(dot(P_best*A*P_best', B))
            P_best = P;
        end
    end