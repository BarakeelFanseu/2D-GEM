% Umeyama's method 
% A and B are the matrices to be matched 
% Return permutation matrix P so that P*A*P' is matched to B 

function [P] = matching_umeyama(A, B)
    n = size(A, 1);
    [U, ~] = eig(A);
    [V, ~] = eig(B);
    X = abs(U) * abs(V)';
    M = matchpairs(X', -99999, 'max');
    P = full(sparse(M(:, 1), M(:, 2), 1, n, n));