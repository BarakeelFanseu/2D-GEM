fprintf('geodesic processing for M...'); 
    M.distances = [];
    vec = linspace(n1/4,n1,4);
    disp(int32(vec))
    begining = 1;
    for i=1:length(vec)%1:n1
        ending = vec(i);
        idx = begining:ending;
        begining = ending + 1;
        distances = perform_fast_marching_mesh(M.VERT', M.TRIV', idx, options.option1);
        M.distances = [M.distances, distances];
        disp(size(M.distances))
    end
    
    fprintf('geodesic processing for N...'); 
    N.distances = [];
    begining = 1;
    for i=1:length(vec)%1:n2
       ending = vec(i);
       idx = begining:ending;
       begining = ending + 1;
       distances = perform_fast_marching_mesh(N.VERT', N.TRIV', idx, options.option2);
       N.distances = [N.distances, distances];
    end