function [N, M, n1, n2, diameters, corr_true] = load_and_preprocess(i,j,path_track, meshes, corres, name, options)
%% Loading M
disp('----------------------------------')
disp(strcat(path_track, meshes, name, i, '.off'))
fprintf('Loading VERT, TRIV... ')

[M.VERT,M.TRIV] = ReadOFF(strcat(name, i, '.off'));
% M = load_off(strcat(path_track, meshes, name, i, '.off'));
M.n = length(M.VERT);
n1=M.n;
fprintf('done \n')



fprintf('preprocessing adj...')
M.adj = digraph(M.TRIV, M.TRIV(:, [2 3 1])); % just use matlab function seems much faster
% plot(M.adj)
M.adj = adjacency(M.adj);
% disp(size(M.adj(M.adj>0), 1))
M.adj = M.adj | M.adj';
% disp(size(M.adj(M.adj>0), 1))
% % disp(M.adj(1:20, 1:20))
% % M.adj = eye(M.n)+ M.adj; %Best GINConv
fprintf('done \n')


fprintf('Computing coltan laplacian... ');        
[W, A] = cotLaplacian(M.VERT, M.TRIV);
W =((W+W')/2);
A = sparse(1:length(A), 1:length(A), A);
W(W==Inf) = 0;
W(W==-Inf) = 0;
M.Phi.A = A;
M.Phi.W = W; 
fprintf('done \n')


%% descriptors for M
diam_shot = sqrt(sum(calc_tri_areas(M)));
M.shots = calc_shot(M.VERT', M.TRIV', 1:n1, options.shot_num_bins, options.shot_radius*diam_shot/100, 3)';

%% Loading N
disp('----------------------------------')

disp(strcat(path_track, meshes, name, j, '.off'))
fprintf('Loading TRIV, VERT ...')

[N.VERT, N.TRIV] = ReadOFF(strcat(name, j, '.off'));
% N = load_off(strcat(path_track, meshes, name, j, '.off'));
N.n = length(N.VERT);
n2=size(N.VERT,1);
fprintf('done.\n')

%% shuffle points of the second shape (will be implemented later)
if (options.isometric)
    if (options.shuffle)
        fprintf('Shuffling TRIV and VERT since isometric... ')
        corr_true = randperm(N.n)'; % shuffle points. Point i in shape 2 corrsponds to point corr_true(i) in shape 1.
        corr_true_reverse(corr_true,1) = 1:N.n; % Point i in shape 1 corrsponds to point corr_true_reverse(i) in shape 2.
        % disp(size(corr_true_reverse))
        % disp(corr_true_reverse(1:10))
    %     corr_true = randperm(N.n)'; % shuffle points. Point i in shape 2 corrsponds to point corr_true(i) in shape 1.
    %     corr_true_reverse(corr_true,1) = 1:N.n; % Point i in shape 1 corrsponds to point corr_true_reverse(i) in shape 2.
        N.VERT = N.VERT(corr_true, :); 
        N.TRIV = corr_true_reverse(N.TRIV);
        fprintf('done.\n')

    else
        fprintf('Isometric but no shuffling used... \n')
        corr_true = [1:N.n]';
        corr_true_reverse(corr_true,1) = 1:N.n; 
    end
elseif ~(options.isometric)
    fprintf('Non-Isometric so no shuffling used... \n')
    corr_true = [1:N.n]';
    corr_true_reverse(corr_true,1) = 1:N.n; 
end

fprintf('preprocessing adj...')
N.adj = digraph(N.TRIV, N.TRIV(:, [2 3 1]));
N.adj = adjacency(N.adj);
N.adj = N.adj | N.adj';
% % N.adj = eye(N.n) + N.adj; %Best GINConv
fprintf('done.\n')

%% Continue loading N
fprintf('Computing coltan laplacian... ');   
[W, A] = cotLaplacian(N.VERT, N.TRIV);
W =((W+W')/2);
A = sparse(1:length(A), 1:length(A), A);
W(W==Inf) = 0;
W(W==-Inf) = 0;
W = sparse(W);
N.Phi.A = A;
N.Phi.W = W;
fprintf('done \n')

diameters = sqrt(sum(calc_tri_areas(N))); 

diam_shot = diameters;

%% N descriptors
N.shots = calc_shot(N.VERT', N.TRIV', 1:n2, options.shot_num_bins, options.shot_radius*diam_shot/100, 3)';
% N.shots = calc_shot(N.VERT', N.TRIV', 1:n2, options.shot_num_bins, options.shot_radius*N.Phi.sqrt_area/100, 3)';


%% diameters
% diameters = N.Phi.sqrt_area; %sqrt(sum(calc_tri_areas(N)));
