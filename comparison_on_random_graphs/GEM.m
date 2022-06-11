% Implementation of our HOPE method 
% A and B are the matrices to be matched 
% X_A and X_B are the feature matrices
% num indicate the number of eigenvalues
% Return permutation matrix P so that P*A*P' is matched to B 

function [P] = GEM(A, B, desc_X, desc_Y, num, power)
    fprintf('Using GEM \n');
    n1 = size(A, 1);
    n2 = size(B, 1);
    
    %% Laplacian A
    Dh_A = diag(ones(n1,1)./sqrt(A*ones(n1,1)));
    Dh_A(Dh_A==Inf) = 0;
    Dh_A(Dh_A==-Inf) = 0;
%     disp(size(Dh_A));
    A = eye(n1) - Dh_A*A*Dh_A;
 
    %% Laplacian B
    Dh_B = diag(ones(n2,1)./sqrt(B*ones(n2,1)));
    Dh_B(Dh_B==Inf) = 0;
    Dh_B(Dh_B==-Inf) = 0;
%     disp(size(Dh_B));
    B = eye(n2) - Dh_B*B*Dh_B;  
    
    
    if nargin < 4
         F1 = ones(n1,n2)';
    else  
        F1 = desc_X*desc_Y';
    end
%     disp(F1(1:5, 1:5))
    
   %% Initializing filter

   if num < n1 || num <n2
       [U, Lambda] = eigs(A, 40, 1e-6); % part of eigen vals
       [V, Mu] = eigs(B, 40, 1e-6); % part
   else
    [U, Lambda] = eig(full(double(A)));
    [V, Mu] = eig(full(double(B)));
   end
   
   lambda = diag(Lambda);
   mu = diag(Mu);
   
   
   if power
       disp('using power')
       Coeff = 1 ./ (((lambda.^10) - (mu.^10)').^2 + 1);
   else
       disp('not using power')
       Coeff = 1 ./ (((lambda) - (mu)').^2 + 1);
   end
   
   %% filteriing
  
   Coeff = Coeff .* (U' * F1 * V);
   Coeff = U * Coeff * V';
   
    %% Rounding by linear assignment - better but slower 
    M = matchpairs(Coeff', -99999, 'max');
    P = full(sparse(M(:, 1), M(:, 2), 1, n1, n1));

    %% Greedy matching - faster but worse 
%     P = full(greedy_match(X'));

    %% Greedy rounding
%     [~, ind_max] = max(X);
%     P = full(sparse(1:n, ind_max, 1, n, n));