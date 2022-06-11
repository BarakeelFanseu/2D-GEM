% IsoRank
% A and B are the matrices to be matched 
% Return permutation matrix P so that P*A*P' is matched to B 

function [P_iso] = matching_isorank(A, B, alpha)
    n = size(A, 1);
    C = kron(B, A);
    P = C ./ sum(C);
%     tic;
%     x = (eye(n^2) - alpha * P)\ones(n^2, 1);
    x = lsqr(eye(n^2) - alpha * P, ones(n^2, 1));
%     toc;
    X = reshape(x, [n, n]);

    M = matchpairs(X', -99999, 'max');
    P_iso = full(sparse(M(:, 1), M(:, 2), 1, n, n));