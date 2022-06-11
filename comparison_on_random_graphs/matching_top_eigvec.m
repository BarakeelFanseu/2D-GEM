% Graph matching by aligning top eigenvectors 
% A and B are the matrices to be matched 
% Return permutation matrix P so that P*A*P' is matched to B 

function [P] = matching_top_eigvec(A, B)
    n = size(A, 1);
    [u, ~] = eigs(A, 1, 'largestreal');
    [v, ~] = eigs(B, 1, 'largestreal');
    [~, ord_u] = sort(u);
    [~, ord_v] = sort(v);
    [~, ord_v2] = sort(-v);
    per(ord_u) = ord_v;
    per2(ord_u) = ord_v2;
    P = full(sparse(per, 1:n, 1, n, n));
    P2 = full(sparse(per2, 1:n, 1, n, n));
    if trace(P*A*P'*B') < trace(P2*A*P2'*B')
        P = P2;
    end