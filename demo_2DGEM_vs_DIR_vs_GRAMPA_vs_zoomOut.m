%% For all shapes this demo implements matching such that M is smaler shape and N is larger shape. 

addpath(genpath('.')),
close all
clear all
clc


%% options
options = struct;
options.shot_num_bins = 10; % number of bins for shot
options.shot_radius = 5;

options.shuffle = false; % leave false as not implemented on different shapes for DIR
options.isometric = false;


options.option1.nb_iter_max = 30; % iteration number needed for fast marching %% this should be smallest
options.option2.nb_iter_max = 120; %% this largest
options.nb_iter_max = inf;


%% loading... for all pairs this demo implements matching larger to smaller shape.
name ='';
% 
%% %%For Isometric uncoment any of the isometric sample pairs below as well as
%% the gt_in line and lines 40-42
% % i = 'mesh023';
% % j = 'mesh071';
% % gt_in = [1:12500]';
% % 
% i ='mesh03';
% j = 'mesh050';
% gt_in = [1:12500]';

% % i ='wolf1';
% % j = 'wolf2';
% % gt_in = [1:4344]';
% 
% options.isometric = true;
% gt = [gt_in, gt_in];
% disp(size(gt))

%% %%for non_isometric uncomment any of the pairs below and the gt_in line
%% as well as lines 58-61
% i = 'kid16'; 
% j = 'kid17';
% gt_in = [1:10988]';%11292

i = 'kid19'; 
j = 'kid17';
gt_in = [1:8515]';%11292

% i = 'kid19'; 
% j = 'kid20';
% gt_in = [1:8515]';%11292

gt_M_null = read_correspondence(strcat(name, i, '_ref.txt'));% load
gt_N_null = read_correspondence(strcat(name, j, '_ref.txt')); %load
gt = merge_ground_truth(gt_M_null, gt_N_null); % merge
disp(size(gt))


%% specs for GEM & DIR
if ~(options.isometric)
    options.maxIter = 60;
    options.spec_dim = 10;
    options.th = [100];
else
    options.maxIter = 10;
    options.spec_dim = 500;    
    low = 0.14; % local distortion lower bound
    gap = 5; % local distortion gaps
    options.th = 0.6-(0.5-low)/gap:-(0.5-low)/gap:low; % local distortion
    disp(options.th)
end

[N, M, n1, n2, diameters, corr_true] = load_and_preprocess(i, j,'', '', '', '', options);
M.surface.X = M.VERT(:,1);
M.surface.Y = M.VERT(:,2);
M.surface.Z = M.VERT(:,3);
M.surface.VERT = M.VERT;
M.surface.TRIV = M.TRIV;
M.nf = size(M.TRIV,1);
M.nv = size(M.VERT,1);

N.surface.X = N.VERT(:,1);
N.surface.Y = N.VERT(:,2);
N.surface.Z = N.VERT(:,3);
N.surface.VERT = N.VERT;
N.surface.TRIV = N.TRIV;
N.nf = size(N.TRIV,1);
N.nv = size(N.VERT,1);

%% GEM
disp('---- GEM preprocess ----')
tic
fprintf('geodesic processing for M...'); 
M.distances = [];
vec = double(int32(linspace(n1/5,n1,5)));
%     disp(vec)
begining = 1;
for kk=1:length(vec)%1:n1
    ending = vec(kk);
    idx = begining:ending;
    begining = ending + 1;
    distances = perform_fast_marching_mesh(M.VERT', M.TRIV', idx, options.option1);
    M.distances = [M.distances, distances];
%         disp(size(M.distances))
end
fprintf('done \n'); 
fprintf('geodesic processing for N...'); 
N.distances = [];
vec = double(int32(linspace(n2/5,n2,5)));
begining = 1;
for kk=1:length(vec)%1:n2
   ending = vec(kk);
   idx = begining:ending;
   begining = ending + 1;
   distances = perform_fast_marching_mesh(N.VERT', N.TRIV', idx, options.option2);
   N.distances = [N.distances, distances];
end
fprintf('done \n');
toc

%% geodesic distances for plot
fprintf('geodesic processing Full N for plot... ');
tic
distances = geodesic_distance(N.TRIV,N.VERT); %Added by JX  
%     distances = perform_fast_marching_mesh(N.VERT', N.TRIV', 1:n2, options);
distances = sparse(distances);
fprintf('done \n')
toc

%% GEM
disp('------GEM------')
tic
corr_GEM = GEM(M, N, corr_true, options);
toc
[corr_GEM, ~] = find(corr_GEM);
% corr = [gt_in, corr_GEM];
figure(1);
subplot(1,2,1); visualize_map_on_source(M, N, corr_GEM); title('Source');
subplot(1,2,2); visualize_map_on_target(M, N, corr_GEM); title('2D-GEM')

%% DIR

options.spec_dim = 500;
options.spec_dim_cut = 460;
if ~(options.isometric)
    options.th = [100];    
end
disp('------DIR------')
tic
corr_DIR = DIR(strcat(i, '.off'), strcat(j, '.off'), options, corr_true);
toc
figure(2);
subplot(1,2,1); visualize_map_on_source(M, N, corr_DIR); title('Source');
subplot(1,2,2); visualize_map_on_target(M, N, corr_DIR); title('DIR')

%% GRAMPA
disp('------GRAMPA------')
tic
corr_GRAMPA = GRAMPA(M, N, options, 1); %eta 1 as in original paper
[corr_GRAMPA, ~] = find(corr_GRAMPA);
toc
figure(3)
subplot(1,2,1); visualize_map_on_source(M, N, corr_GRAMPA); title('Source');
subplot(1,2,2); visualize_map_on_target(M, N, corr_GRAMPA); title('GRAMPA')

%% ZoomOut
options.spec_dim = 200;
disp('------ZoomOut------')
options.k_init = 20;
options.k_step = 1;
options.k_final = 200; % as in original paper from 4 or 20 to a max of 200 
tic
[corr_ZoomOut, ~, ~, ~] = zoomOut_refine(M, N, options);
toc
figure(4)
subplot(1,2,1); visualize_map_on_source(M, N, corr_ZoomOut); title('Source');
subplot(1,2,2); visualize_map_on_target(M, N, corr_ZoomOut); title('ZoomOut')

%% getting curves
all_corr = cell(4,1);
all_corr{1} = [gt_in, corr_GEM];
all_corr{2} = [gt_in, corr_GRAMPA];
all_corr{3} = [gt_in, corr_DIR];
all_corr{4} = [gt_in, corr_ZoomOut];

all_curves = cell(4,1);
for i=1:4
    corr = cell2mat(all_corr(i));
%     disp(size(corr))
    errors = zeros(size(corr,1), 1);
    for m=1:size(corr,1)  
        gt_match = gt(gt(:,1) == corr(m,1), 2);
        match = corr(m,2);
        
        if ~isempty(gt_match)                                                    
            % using the geodesic distance of the second graph
            errors(m) = distances(gt_match, match); % TODO include your geodesics here
        else
            errors(m) = 200;
        end
    end
    thresholds = 0:0.01:0.25;
    errors = errors / diameters;
    curve = zeros(1, length(thresholds));
    for m=1:length(thresholds)
        curve(m) = 100*sum(errors <= thresholds(m)) / length(errors);
    end
    all_curves{i} = curve;
end


%% combined
figure(5);
subplot(1,6,1); visualize_map_on_source(M, N, corr_GEM); title('Source');
subplot(1,6,2); visualize_map_on_target(M, N, corr_GEM); title('GEM')
subplot(1,6,3); visualize_map_on_target(M, N, corr_GRAMPA); title('GRAMPA');
subplot(1,6,4); visualize_map_on_target(M, N, corr_DIR); title('DIR')
subplot(1,6,5); visualize_map_on_target(M, N, corr_ZoomOut); title('ZoomOut')

figure(7)
plot(thresholds', mean(all_curves{1}, 1)', thresholds', mean(all_curves{2}, 1)', thresholds', mean(all_curves{3}, 1)', thresholds', mean(all_curves{4}, 1)'), 
ylim([0 100]), 
line_width=1.5;
hline = findobj(gcf, 'type', 'line');
set(hline,'LineWidth',line_width);
legend({'2D-GEM', 'GRAMPA', 'DIR', 'ZoomOut'},'FontSize', 10);


