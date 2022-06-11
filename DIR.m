function pertF = DIR(name1, name2, options, corr_true)

addpath(genpath('.'));
spec_dim = options.spec_dim; % max spectrum dimension
spec_dim_cut = options.spec_dim_cut; 
option1.nb_iter_max = 30; % iteration number needed for fast marching
option2.nb_iter_max = 120;
th = options.th;
iter_number = options.maxIter;

%%
[surf2.pt,surf2.trg] = ReadOFF(name2);
S2 = MESH.MESH_IO.read_shape(name2);
surf2.Phi = MESH.compute_LaplacianBasis(S2, spec_dim);
surf2.n = length(surf2.pt);
num2 = surf2.n;
opts.shot_num_bins = 10; % number of bins for shot
opts.shot_radius = 5; % percentage of the diameter used for shot
Xdesc = calc_shot(surf2.pt', surf2.trg', 1:num2, opts.shot_num_bins, opts.shot_radius*surf2.Phi.sqrt_area/100, 3)';
vertex2 = surf2.pt';
faces2 = surf2.trg';
    
[surf1.pt,surf1.trg] = ReadOFF(name1);
S1 = MESH.MESH_IO.read_shape(name1);

surf1.Phi = MESH.compute_LaplacianBasis(S1, spec_dim);  
surf1.n = length(surf1.pt);
MA = full(diag(surf1.Phi.A));
xdesc = calc_shot(surf1.pt', surf1.trg', 1:surf1.n, opts.shot_num_bins, opts.shot_radius*surf1.Phi.sqrt_area/100, 3)';
pertF = knnsearch(Xdesc, xdesc,'NSMethod','kdtree');

if surf1.n < surf2.n
    num = surf1.n;
else
    num = surf2.n;
end
vertex1 = surf1.pt';
faces1 = surf1.trg';
cnt = 0;
e = zeros(num,1);
D1 = perform_fast_marching_mesh(vertex1, faces1, 1:surf1.n, option1);
D2 = perform_fast_marching_mesh(vertex2, faces2, 1:surf2.n, option2);
R_max = max(max(D2));
landmarks = [];
for kk = 1:iter_number
    DD1 = cell(surf1.n,1);
    DD2 = cell(surf2.n,1);
    idx = cell(num,1);    
    ee = 0;
    good = 1:num;
    D1T = D1(corr_true(good),good);
    D2T = D2(good,pertF(good));
    for i = 1:length(good)
        idx{i} = find(D1T(:,i) ~= 0);
        DD1{i} = D1T(idx{i},i);
        DD2{i} = D2T(idx{i},i);
        DD2{i}(DD2{i}==0) = R_max;
        r = max(DD1{i});
        ee(i) = sum(((abs(DD1{i}-DD2{i}))/r).*MA(idx{i}))/sum(MA(idx{i}));
    end
    e(good)=ee;
    if kk<=length(th)
        landmarks = find(e<th(kk));
    else
        landmarks = find(e<th(end));
    end
    sub_landmarks = landmarks;
    good = setdiff(1:num,sub_landmarks);
    goodF = setdiff(1:num,pertF(sub_landmarks));
    H = surf1.Phi.evecs(sub_landmarks,:)'*surf2.Phi.evecs(pertF(sub_landmarks),:);
    [U,D,V] = svd(H);
    dim = findK(diag(D));
    if dim>spec_dim_cut
        cnt = cnt + 1;
    end
    if cnt>3
        break;
    end
    specB = surf2.Phi.evecs(:,1:dim);
    specA = surf1.Phi.evecs(:,1:dim);
    H = specA(sub_landmarks,:)'*specB(pertF(sub_landmarks),:);
    [U,D,V] = svd(H);
    C = U*V';
    specBB = specB(goodF,:);
    specAA = specA(good,:);
    p = knnsearch(specBB*C',specAA,'NSMethod','kdtree');
    pertF(good) = goodF(p);
end
