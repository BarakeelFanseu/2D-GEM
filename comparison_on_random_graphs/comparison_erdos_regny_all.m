%Comparing_Erdos_Regny_Random model
clc
%% initialization
p = 0.5;
num_run = 1;
%vec_dim = 3:3:3;
vec_dim = 100:100:100;
%vec_dim = [100,200,400,800,1000];
len_dim = length(vec_dim);
vec_noise = 0:0.05:0.8;
len_noise = length(vec_noise);
feats_type = 'non-dis'; % non-dis == non-distinguishable, dist == distinguishable, rand == random
percentage_dis =1; % determines the percentage of distinguishable features 
num_eigs = 100;
use_power = true;

% disp(1e-2);
% disp(1e2)

%% runtimes for plots
topeig_run_forplot = zeros(len_dim, 1);%, len_noise, num_run);
isorank_run_forplot = zeros(len_dim, 1);%, len_noise, num_run);
eigalign_run_forplot = zeros(len_dim, 1);%, len_noise, num_run);
lowrank_run_forplot = zeros(len_dim, 1);%, len_noise, num_run);
umeyama_run_forplot = zeros(len_dim, 1);%, len_noise, num_run);
robust_run_forplot = zeros(len_dim, 1);%, len_noise, num_run);
GEM_run_forplot = zeros(len_dim, 1);%, len_noise, num_run);
full_qp_run_forplot = zeros(len_dim, 1);%, len_noise, num_run);
deg_pro_run_forplot = zeros(len_dim, 1);%, len_noise, num_run);




%% Iteration over dimensions 
for ind_dim = 1:len_dim
    fprintf('different sizes are: '); disp(vec_dim);
    n = vec_dim(ind_dim);
    fprintf('Matrix dimension %i \n', n);

    %% results_corr
    topeig_corr = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    isorank_corr = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    eigalign_corr = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    lowrank_corr = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    umeyama_corr = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    robust_corr = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    GEM_corr = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    full_qp_corr = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    deg_pro_corr = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);

    %% run_times
    topeig_run = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    isorank_run = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    eigalign_run = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    lowrank_run = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    umeyama_run = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    robust_run = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    GEM_run = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    full_qp_run = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    deg_pro_run = zeros(1, len_noise, num_run); %zeros(len_dim, len_noise, num_run);
    
    %% corr for plots
    %topeig_corr_forplot = zeros(num_run, 1); %= topeig_corr_mean;
    %isorank_corr_forplot = zeros(num_run, 1); %= isorank_corr_mean;
    %eigalign_corr_forplot = zeros(num_run, 1); %= eigalign_corr_mean;
    %lowrank_corr_forplot = zeros(num_run, 1); %= lowrank_corr_mean;
    %umeyama_corr_forplot = zeros(num_run, 1); %= umeyama_corr_mean;
    %robust_corr_forplot = zeros(num_run, 1); %= robust_corr_mean;
    %GEM_corr_forplot = zeros(num_run, 1); %= GEM_corr_mean;
    %full_qp_corr_forplot = zeros(num_run, 1); %= full_qp_corr_mean;
    %deg_pro_corr_forplot = zeros(num_run, 1); %= deg_pro_corr_mean;


    %% Iteration over independent samples 
    for ind_run = 1:num_run
        fprintf('Iteration %i \n', ind_run);
        
        %parent graph
        A=binornd(1,p,n,n);
        A=triu(A,1);
        A=A+A';

        %% Iteration over noise levels 
        for ind_noise = 1:len_noise
            sigma = vec_noise(ind_noise); fprintf('sigma: '); disp(sigma);
            s=1-sigma^2;
            
            %% INFO
            %{
            s = 1 - sigma^2 in page 14 of the degree profile paper@https://arxiv.org/pdf/1811.07821.pdf 
            see also https://github.com/xjmoffside/degree_profile/blob/master/DP_comparison/ER_comparison.m
            %}
            
            %% subsample graph
            Z1=binornd(1,s,n,n);
            Z1=triu(Z1,1);
            Z1=Z1+Z1';
            A1=A.*Z1;
            
            perp_rnd=randperm(n);
            %perp_rnd=[1:1:n];
            P_rnd=zeros(n,n);
            P_rnd(1:1:n,perp_rnd)=eye(n);
            A_permuted= P_rnd*A*P_rnd';

            Z2=binornd(1,s,n,n);
            Z2=triu(Z2,1);
            Z2=Z2+Z2';
            A2=A_permuted.*Z2;
            
            W1=A1;
            W2=A2;
            
            
            if strcmp(feats_type, 'rand')
                % random feats for GEM
                disp('random features')
                W1Ft = randi([0,1], [n,10]); % random feats
            elseif strcmp(feats_type, 'non-dis')
                % indistinguishable feats
                disp('indistinguishable feats')
                W1Ft = ones(n); % indistinguishable feats
            else
                 fprintf('i% distinguishable feats \n', percentage_dis*100)
            %   distinguishable feas
%               W1Ft = eye(n); % fully distinguishable feats
            
                num = floor(n*percentage_dis);
                W1Ft = ones(n, num); 
                tenpercentdist = eye(floor(num));
                indxforswitch = randperm(100, num);
                W1Ft(indxforswitch, :) = tenpercentdist;
            
                disp(indxforswitch)
                disp(size(W1Ft))
%                 disp(W1Ft(indxforswitch, :))
            end
            
            %% other graph's feats
            W2Ft = P_rnd*W1Ft;
            
            
            %% Top eigenvector alignment 
            tic;
            P = matching_top_eigvec(W1, W2);
            run_time = toc;
            fix_pt_ratio = sum(dot(P_rnd, P)) / n;
            topeig_corr(1, ind_noise, ind_run) = fix_pt_ratio;
            %topeig_corr(ind_dim, ind_noise, ind_run) = fix_pt_ratio;
            %run_time = toc;
            %topeig_run(ind_dim, ind_noise, ind_run) = run_time;
            topeig_run(1, ind_noise, ind_run) = run_time;
            
            %% IsoRank
            tic;
            P = matching_isorank(W1, W2, 0.85);
            run_time = toc;
            fix_pt_ratio = sum(dot(P_rnd, P)) / n;
            %isorank_corr(ind_dim, ind_noise, ind_run) = fix_pt_ratio;
            %isorank_run(ind_dim, ind_noise, ind_run) = run_time;
            isorank_corr(1, ind_noise, ind_run) = fix_pt_ratio;
            isorank_run(1, ind_noise, ind_run) = run_time;
            
            %% EigenAlign
            tic;
            P = matching_eigenalign(W1, W2, 0.1/sqrt(n*p*(1-p)));
            run_time = toc;
            fix_pt_ratio = sum(dot(P_rnd, P)) / n;
            %eigalign_corr(ind_dim, ind_noise, ind_run) = fix_pt_ratio;
            %eigalign_run(ind_dim, ind_noise, ind_run) = run_time;
            eigalign_corr(1, ind_noise, ind_run) = fix_pt_ratio;
            eigalign_run(1, ind_noise, ind_run) = run_time;
            
            %% LowRankAlign
            tic;
            P = matching_lowrankalign(W1, W2, 2);
            run_time = toc;
            fix_pt_ratio = sum(dot(P_rnd, P)) / n;
            %lowrank_corr(ind_dim, ind_noise, ind_run) = fix_pt_ratio;
            %lowrank_run(ind_dim, ind_noise, ind_run) = run_time;
            lowrank_corr(1, ind_noise, ind_run) = fix_pt_ratio;
            lowrank_run(1, ind_noise, ind_run) = run_time;
            
            %% Umeyama's method 
            tic;
            P = matching_umeyama(W1, W2);
            run_time =toc;
            fix_pt_ratio = sum(dot(P_rnd, P)) / n;
            %umeyama_corr(ind_dim, ind_noise, ind_run) = fix_pt_ratio;
            %umeyama_run(ind_dim, ind_noise, ind_run) = run_time;
            umeyama_corr(1, ind_noise, ind_run) = fix_pt_ratio;
            umeyama_run(1, ind_noise, ind_run) = run_time;
            
            %% Robust spectral method 
            tic;
            P = matching_robust_spectral(W1, W2, 0.2);
            run_time = toc;
            fix_pt_ratio = sum(dot(P_rnd, P)) / n;
            %robust_corr(ind_dim, ind_noise, ind_run) = fix_pt_ratio;
            %robust_run(ind_dim, ind_noise, ind_run) = run_time;
            robust_corr(1, ind_noise, ind_run) = fix_pt_ratio;
            robust_run(1, ind_noise, ind_run) = run_time;
                        
            %% k-hop method
            tic;
            
            P = GEM(W1, W2, W1Ft, W2Ft, num_eigs, use_power);
            run_time = toc;
            fix_pt_ratio = sum(dot(P_rnd, P)) / n;
            %GEM_corr(ind_dim, ind_noise, ind_run) = fix_pt_ratio;
            %GEM_run(ind_dim, ind_noise, ind_run) = run_time;
            GEM_corr(1, ind_noise, ind_run) = fix_pt_ratio;
            GEM_run(1, ind_noise, ind_run) = run_time;
            
            %% Full QP 
            tic;
            P = matching_full_qp(W1, W2);
            run_time = toc;
            fix_pt_ratio = sum(dot(P_rnd, P)) / n;
            %full_qp_corr(ind_dim, ind_noise, ind_run) = fix_pt_ratio;
            %full_qp_run(ind_dim, ind_noise, ind_run) = run_time;
            full_qp_corr(1, ind_noise, ind_run) = fix_pt_ratio;
            full_qp_run(1, ind_noise, ind_run) = run_time;
            
            %% Degree profile
            tic;
            P = matching_deg_pro(W1, W2);
            run_time = toc;
            fix_pt_ratio = sum(dot(P_rnd, P)) / n;
            %deg_pro_corr(ind_dim, ind_noise, ind_run) = fix_pt_ratio;
            %deg_pro_run(ind_dim, ind_noise, ind_run) = run_time;
            deg_pro_corr(1, ind_noise, ind_run) = fix_pt_ratio;
            deg_pro_run(1, ind_noise, ind_run) = run_time;
            
        end
    end
    %% mean_correlations
    topeig_corr_mean = mean(topeig_corr, 3);
    isorank_corr_mean = mean(isorank_corr, 3);
    eigalign_corr_mean = mean(eigalign_corr, 3);
    lowrank_corr_mean = mean(lowrank_corr, 3);
    umeyama_corr_mean = mean(umeyama_corr, 3);
    robust_corr_mean = mean(robust_corr, 3);
    deg_pro_corr_mean = mean(deg_pro_corr, 3);
    full_qp_corr_mean = mean(full_qp_corr, 3);
    GEM_corr_mean = mean(GEM_corr, 3);
    
    %fprintf('[topeig_corr_mean, isorank_corr_mean, eigalign_corr_mean, lowrank_corr_mean, umeyama_corr_mean, robust_corr_mean, deg_pro_corr_mean, full_qp_corr_mean, GEM_corr_mean] Correlations are: \n');
    %disp([topeig_corr_mean, isorank_corr_mean, eigalign_corr_mean, lowrank_corr_mean, umeyama_corr_mean, robust_corr_mean, deg_pro_corr_mean, full_qp_corr_mean, GEM_corr_mean]);
    
    %% mean times
    topeig_run_mean = mean(mean(mean(topeig_run(1, 3:end, :)))); %mean(topeig_run, 3);
    isorank_run_mean = mean(mean(mean(isorank_run(1, 3:end, :))));
    eigalign_run_mean = mean(mean(mean(eigalign_run(1, 3:end, :))));
    lowrank_run_mean = mean(mean(mean(lowrank_run(1, 3:end, :))));
    umeyama_run_mean = mean(mean(mean(umeyama_run(1, 3:end, :))));
    robust_run_mean = mean(mean(mean(robust_run(1, 3:end, :))));
    deg_pro_run_mean = mean(mean(mean(deg_pro_run(1, 3:end, :))));
    full_qp_run_mean = mean(mean(mean(full_qp_run(1, 3:end, :))));
    GEM_run_mean = mean(mean(mean(GEM_run(1, 3:end, :))));
    
    fprintf('[topeig_run_mean, isorank_run_mean, eigalign_run_mean, lowrank_run_mean, umeyama_run_mean, robust_run_mean, deg_pro_run_mean, full_qp_run_mean, GEM_run_mean] Correlations are: \n');
    disp([topeig_run_mean, isorank_run_mean, eigalign_run_mean, lowrank_run_mean, umeyama_run_mean, robust_run_mean, deg_pro_run_mean, full_qp_run_mean, GEM_run_mean]);
    
    
    %% Runs For plots
    topeig_run_forplot(ind_dim, 1) = topeig_run_mean;
    isorank_run_forplot(ind_dim, 1) = isorank_run_mean;
    eigalign_run_forplot(ind_dim, 1) = eigalign_run_mean;
    lowrank_run_forplot(ind_dim, 1) = lowrank_run_mean;
    umeyama_run_forplot(ind_dim, 1) = umeyama_run_mean;
    robust_run_forplot(ind_dim, 1) = robust_run_mean;
    GEM_run_forplot(ind_dim, 1) = GEM_run_mean;
    full_qp_run_forplot(ind_dim, 1) = full_qp_run_mean;
    deg_pro_run_forplot(ind_dim, 1) = deg_pro_run_mean;
    
    %% Corr for Plots
    %topeig_corr_forplot(ind_dim, 1) = topeig_corr_mean;
    %isorank_corr_forplot(ind_dim, 1) = isorank_corr_mean;
    %eigalign_corr_forplot(ind_dim, 1) = eigalign_corr_mean;
    %lowrank_corr_forplot(ind_dim, 1) = lowrank_corr_mean;
    %umeyama_corr_forplot(ind_dim, 1) = umeyama_corr_mean;
    %robust_corr_forplot(ind_dim, 1) = robust_corr_mean;
    %GEM_corr_forplot(ind_dim, 1) = GEM_corr_mean;
    %full_qp_corr_forplot(ind_dim, 1) = full_qp_corr_mean;
    %deg_pro_corr_forplot(ind_dim, 1) = deg_pro_corr_mean;
    
    
    %% Plotting corr
    line_width = 1.5;
    Marker_size=6;
    plot_spec={'k-o','c-o', 'r-o','m--*','b-o','b--+', 'g--*', 'g-o', 'r--*'};
    leng_spec = {'topeig','isorank','eigalign','lowrank','GRAMPA','2D-GEM', 'QP', 'DP', 'Umeyama'};
    figure;
    plot(vec_noise, topeig_corr_mean, plot_spec{1},'LineWidth', line_width, 'MarkerSize', Marker_size );
    hold on;
    plot(vec_noise, isorank_corr_mean, plot_spec{2},'LineWidth', line_width, 'MarkerSize', Marker_size );
    hold on;
    plot(vec_noise, eigalign_corr_mean, plot_spec{3},'LineWidth', line_width, 'MarkerSize', Marker_size );
    hold on;
    plot(vec_noise, lowrank_corr_mean, plot_spec{4},'LineWidth', line_width, 'MarkerSize', Marker_size );
    hold on;
    plot(vec_noise, robust_corr_mean, plot_spec{5},'LineWidth', line_width, 'MarkerSize', Marker_size );
    hold on;
    plot(vec_noise, GEM_corr_mean, plot_spec{6},'LineWidth', line_width, 'MarkerSize', Marker_size );
    hold on;
    plot(vec_noise, full_qp_corr_mean, plot_spec{7},'LineWidth', line_width, 'MarkerSize', Marker_size );
    hold on;
    plot(vec_noise, deg_pro_corr_mean, plot_spec{8},'LineWidth', line_width, 'MarkerSize', Marker_size );
    hold on;
    plot(vec_noise, umeyama_corr_mean, plot_spec{9},'LineWidth', line_width, 'MarkerSize', Marker_size );
    
    legend(leng_spec,'location', 'best', 'FontSize', 20,'Interpreter','latex');
    xlabel('$\sigma$','FontSize',20,'Interpreter','latex');
    ylabel ('fraction of correctly matched pairs','FontSize',20,'Interpreter','latex');
    xlim([0 0.8])
    savefilename = strcat('comparison');%, int2str(n));
    saveas(gcf, savefilename, 'fig');
end

%% saving_run
%clear -regexp _corr$ _run$;
save(strcat('.\mat_files\', savefilename,'.mat'));