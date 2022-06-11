function [pert] = GEM(surf1, surf2, corr_true, options)

%% initialize 2-hop
n1 = surf1.n;
n2 = surf2.n;
EYE1=sparse(1:n1,1:n1,1,n1,n1);
EYE2=sparse(1:n2,1:n2,1,n2,n2);
W21=sparse((double((surf1.adj*surf1.adj)>0)-surf1.adj-EYE1)>0);
W22=sparse((double((surf2.adj*surf2.adj)>0)-surf2.adj-EYE2)>0);

%% initialize shots
pert = surf2.shots*surf1.shots';   
pert = greedy_match(pert);

%% initialize filter
[~, U, lambda] = laplacian_from_TRIV_adj(surf2.adj, options.spec_dim);
[~, V, mu] = laplacian_from_TRIV_adj(surf1.adj, options.spec_dim);
Coeff = 1 ./ ((lambda - mu').^2 + 1);

%% local map improvement 
[num,~] = find(pert);
num = size(num,1);
MA = full(diag(surf1.Phi.A));
e = zeros(num,1);
R_max = max(max(surf2.distances));
for kk = 1:options.maxIter
    pertF = greedy_match(W22*pert*W21);
    [pertF,col] = find(pertF);       
    DD1 = cell(surf1.n,1);
    DD2 = cell(surf2.n,1);
    idx = cell(num,1);
    ee = zeros(num,1);
    good = 1:num;
%         corr_true = good;
    D1T = surf1.distances(corr_true(good),good);  %surf1.distances(corr_true,good); shuffiling not implemented for this demo
    D2T = surf2.distances(good,pertF(good));
    for i = 1:length(good)
        idx{i} = find(D1T(:,i) ~= 0);
        DD1{i} = D1T(idx{i},i);
        DD2{i} = D2T(idx{i},i);
        DD2{i}(DD2{i}==0) = R_max;
        r = max(DD1{i});
        ee(i) = sum(((abs(DD1{i}-DD2{i}))/r).*MA(idx{i}))/sum(MA(idx{i}));        
    end
    e(good)=ee; 
    if kk<=length(options.th)
        landmarks = find(e<options.th(kk));
    else
        landmarks = find(e<options.th(end));
    end
    sub_landmarks = landmarks;
%         disp(size(landmarks))%%          
    
    good = setdiff(1:num,sub_landmarks);
    goodF = setdiff(1:num,pertF(sub_landmarks));         
    p =  U(pertF(sub_landmarks),:)'*V(sub_landmarks,:); 
    p = Coeff .* p;
    p = U(goodF, :) * p * V(good, :)';
    [p,] = greedy_match(p);
    [p, ~] = find(p);
    pertF(good) = goodF(p);
    pert = sparse(pertF, col, 1, surf2.n, surf1.n);
end

