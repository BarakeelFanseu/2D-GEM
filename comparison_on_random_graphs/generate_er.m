function [A, B, A0, B0, P_rnd] = generate_er(n, p, sigma)
    s = 1 - sigma^2 * (1-p);
    G = rand(n) < p/s;
    G = tril(G, -1) + tril(G, -1)';
    Z1 = rand(n) < s;
    Z1 = tril(Z1, -1) + tril(Z1, -1)';
    Z2 = rand(n) < s;
    Z2 = tril(Z2, -1) + tril(Z2, -1)';
    A0 = G .* Z1;
    B0 = G .* Z2;

    P_rnd = eye(n);
    P_rnd = P_rnd(:, randperm(n));
    B0 = P_rnd * B0 * P_rnd';

    A = A0 - p;
    B = B0 - p;
    A = A/sqrt(n*p*(1-p));
    B = B/sqrt(n*p*(1-p));