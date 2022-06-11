clear all
close all

M.adj = [0 1 0 1; 1 0 1 0; 0 1  0 1; 1 0 1 0];
disp('adj')
disp(M.adj)

% % N.adj = [0 1 0 1; 1 0 1 0; 0 1  0 1; 1 0 1 0];

n1 = size(M.adj, 1);

EYE1=sparse(1:n1,1:n1,1,n1,n1);
% % EYE2=sparse(1:n2,1:n2,1,n2,n2);
% W21=double((double((M.adj*M.adj)>0)-EYE1)>0);
% % W22=double((double((N.adj*N.adj)>0)-N.adj-EYE2)>0);
% disp(W21)

W21 = M.adj;
for i=1:1:4
    W21 = W21*M.adj;
% %     W22 = N.adj*N.adj;
end

disp('test1')
W21 = double((double((W21)>0)-M.adj-EYE1)>0);
% % W22 = double((double((W22)>0)-N.adj-EYE2)>0);
disp(W21)

disp('test2')
W21 = double((double((W21)>0)-EYE1)>0);
disp(W21)