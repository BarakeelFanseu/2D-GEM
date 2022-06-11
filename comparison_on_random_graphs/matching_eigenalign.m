% EigenAlign
% A and B are the matrices to be matched 
% Return permutation matrix P so that P*A*P' is matched to B 

function [P] = matching_eigenalign(A, B, gamma)
    n = size(A, 1);
    J = ones(n);
    C = kron(B - gamma*J, A - gamma*J);
    [x, ~] = eigs(C, 1, 'largestreal');
    X = reshape(x, [n, n]);

    M1 = matchpairs(X', -99999, 'max');
    P = full(sparse(M1(:, 1), M1(:, 2), 1, n, n));
    M2 = matchpairs(X', 99999, 'min');
    P2 = full(sparse(M2(:, 1), M2(:, 2), 1, n, n));
    if sum(dot(P*A*P', B)) < sum(dot(P2*A*P2', B))
        P = P2;
    end