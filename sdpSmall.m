function [G, Constraint, G_v, A, X_p, X_m, w] = sdpSmall(epsilon, mu, probs, r_sphere, r_slab, opts, M, y, h, num_x)
    nc = size(M,1);
    assert(size(y,1) == nc);
    assert(size(h,1) == nc);
    sizeG = 5 + nc;
    Gxw = sdpvar(3);
    Gs = sdpvar(3, sizeG-3, 'full');
    Mo = [mu'; M];
    % inner product of known terms
    Go = Mo * Mo';
    G = [Gxw Gs; Gs' Go];
    Gc = [Gxw Gs(1:3,1:2); Gs(1:3,1:2)' Go(1:2,1:2)];
    i_x = @(i) i;
    i_w = 3;
    i_mu = @(i) i+3;
    i_cons = @(i) i+5;
    
    Constraint = [G >= 0];
    % <x_p, w> >= 0.5
    % <x_m, w> <= -0.5
    Constraint = [Constraint;
                   0.6 <= G(i_x(1),i_w) <= 1.0;
                  -1.0 <= G(i_x(2),i_w) <= -0.6];
    
    % M_p * x_p <= y_p
    % M_m * x_m <= y_m
    % M_w * w <= y_w
    for i=1:nc
        if h(i) == 3
            Constraint = [Constraint;
                          G(h(i), i_cons(i)) ...
                          - epsilon * probs(1) * G(i_w, i_x(1)) ...
                          + epsilon * probs(2) * G(i_w, i_x(2)) <= y(i)];
        else
            Constraint = [Constraint;
                          G(h(i), i_cons(i)) <= y(i)];
        end
    end
    
    % constrain \ell_2 norm to class center
    mean_vec = [epsilon/(1+epsilon); -epsilon/(1+epsilon); 0; 1/(1+epsilon); -1/(1+epsilon)];
    for i=1:2
        v = zeros(5,1);
        v(i_x(i)) = 1;
        v(i_mu(i)) = -1;
        Constraint = [Constraint;
                      v' * Gc * v <= r_sphere(i)^2];

        % constrain attack to be orthogonal to mean
        Constraint = [Constraint;
                      v' * Gc * mean_vec <= r_slab(i);
                      v' * Gc * mean_vec >= -r_slab(i)];
    end
              
              
    Objective = [probs(1) -probs(2) 0 0 0]*Gc*[0; 0; 0; 1; -1];
    
    optimize(Constraint, Objective, opts);
    fprintf(1, 'Objective: %.4f\n', double(Objective));
    G_v = double(G);
    
    d = size(mu,1);
    
    X_p = zeros(d, num_x);
    X_m = zeros(d, num_x);
    for i=1:num_x
        x0_p = randn(d,1);
        x0_m = randn(d,1);
        w0 = randn(d,1);

        A_0 = [x0_p x0_m w0 mu M'];
        G_0 = A_0' * A_0;
        G_0 = G_0(sizeG:-1:1,sizeG:-1:1);
        G_v = G_v(sizeG:-1:1,sizeG:-1:1);
        R_0 = chol(G_0 + 1e-6 * eye(sizeG));
        R_v = chol(G_v + 1e-6 * eye(sizeG));
        R = R_0\R_v;
        R = R(sizeG:-1:1,sizeG:-1:1);
        G_0 = G_0(sizeG:-1:1,sizeG:-1:1);
        G_v = G_v(sizeG:-1:1,sizeG:-1:1);
        A = A_0 * R;
        X_p(:,i) = A(:,1);
        X_m(:,i) = A(:,2);
        w = A(:,3);
    end
end