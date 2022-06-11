function [T12, C21, all_T12, all_C21] = zoomOut_refine(M, N, options)
try
    [evecs, evals] = eigs(M.Phi.W, M.Phi.A, options.spec_dim, 1e-6);
catch
    % In case of trouble make the laplacian definite
    [evecs, evals] = eigs(M.Phi.W - 1e-8*speye(M.n), M.Phi.A, options.spec_dim, 'sm');
end
evals = diag(evals);
if ~isreal(evecs)
    evecs(1:2:end) = real(evecs(1:2:end));
    evecs(2:2:end) = imag(evecs(1:2:end));
end
[evals, order] = sort(abs(evals),'ascend');
evecs = evecs(:,order);
B1_all = evecs;

try
    [evecs, evals] = eigs(N.Phi.W, N.Phi.A, options.spec_dim, 1e-6);
catch
    % In case of trouble make the laplacian definite
    [evecs, evals] = eigs(N.Phi.W - 1e-8*speye(N.n), N.Phi.A, options.spec_dim, 'sm');
end
evals = diag(evals);
if ~isreal(evecs)
    evecs(1:2:end) = real(evecs(1:2:end));
    evecs(2:2:end) = imag(evecs(1:2:end));
end
[evals, order] = sort(abs(evals),'ascend');
evecs = evecs(:,order);
B2_all = evecs;

% T12 = N.shots*M.shots';   
% T12 = greedy_match(T12);
% [T12, ~] = find(T12);
T12 = knnsearch(N.shots, M.shots,'NSMethod','kdtree');

if nargout > 2, all_T12 = {}; all_C21 = {}; end

for k = options.k_init : options.k_step : options.k_final
    B1 = B1_all(:, 1:k);
    B2 = B2_all(:, 1:k);
    C21 = B1\B2(T12,:);
    T12 = knnsearch(B2*C21', B1);

    if nargout > 2, all_T12{end+1} = T12; all_C21{end+1} = C21;

end

end