function [G, Constraint, G_v, A, X_p, X_m, W] = sdpSmall(epsilon, mu, probs, r_sphere, r_slab, opts, M, y, h, c, num_x)
    nc = size(M,1);
    assert(size(y,1) == nc);
    assert(size(h,1) == nc);
    sizeG = 5 + nc;
    Gxw = sdpvar(3);
    Gs = sdpvar(3, sizeG-3, 'full');
    Mo = [mu'; diag(c) \ M];
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
                   0.6 <= G(i_x(1),i_w) <= 0.95;
                  -0.95 <= G(i_x(2),i_w) <= -0.6];
              
    % TODO: consider adding constraint that w misclassifies the mean
    
    % M_p * x_p <= y_p
    % M_m * x_m <= y_m
    % M_w * w <= y_w
    i=1;
    Objective = -100;
    
    while i <= nc
        assert(h(i) > 0);
        if h(i) == 3
%             Constraint = [Constraint;
%                           G(h(i), i_cons(i)) ...
%                           - epsilon * probs(1) * G(i_w, i_x(1)) / c(i) ...
%                           + epsilon * probs(2) * G(i_w, i_x(2)) / c(i) <= y(i) / c(i)];
            lossBound = c(i) * G(i_w, i_cons(i));
            lossBound = lossBound - epsilon * probs(1) * G(i_w, i_x(1));
            lossBound = lossBound + epsilon * probs(2) * G(i_w, i_x(2));
            slack = -c(i+1) * epsilon * probs(1) * G(i_x(1), i_cons(i+1));
            slack = slack + c(i+1) * epsilon * probs(2) * G(i_x(2), i_cons(i+1));
            %Objective = max(Objective, -c(i) * G(i_w, i_cons(i)) + y(i+1));
            %Constraint = [Constraint; lossBound - y(i+1) <= 1];
            Constraint = [Constraint;
                          lossBound - slack <= y(i)];
            i = i+2;
        else
            precond = 1; %max(c(i), y(i));
            Constraint = [Constraint;
                          (c(i) / precond) * G(h(i), i_cons(i)) <= (y(i) / precond)];
            i = i+1;
        end
    end
    
    if true || sum(h==3) == 0
        Objective = [probs(1) -probs(2) 0 0 0]*Gc*[0; 0; 0; 1; -1];
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
              
              
    
    optimize(Constraint, Objective, opts);
    fprintf(1, 'Objective: %.4f\n', double(Objective));
    G_v = double(G);
    G_v = (G_v + G_v')/2;
    assert(issymmetric(G_v));
    
    d = size(mu,1);
    
    X_p = zeros(d, num_x);
    X_m = zeros(d, num_x);
    W = zeros(d, num_x);
    dbstop if error;
%    [Proj_half, ~] = qr([mu M'], 0);
    [Proj_half, ~] = svd_lr(Mo', 1e-6);
    G_v = psd_proj(G_v);
    G_11 = G_v(1:3,1:3);
    G_12 = G_v(1:3,4:end);
    G_22 = G_v(4:end,4:end);
%    G_22_pinv = pinv(G_22);
    [U_22, D_22] = eig_lr(G_22, 1e-6);
    Gp_11 = G_11;
    Gp_12 = G_12 * U_22;
    Gp_22 = D_22; %U_22' * G_22 * U_22;
%    G_22_sqrt = U_22 * sqrt(D_22);
    Gp_22_pinv_sqrt = sqrt(inv(D_22)); %U_22 * sqrt(inv(D_22));
    Gp_schur_sqrt = Gp_12 * Gp_22_pinv_sqrt;
    Bp = Gp_schur_sqrt * Gp_22_pinv_sqrt';
    AAt = (G_11 - (Gp_schur_sqrt * Gp_schur_sqrt'));
    [U_a, D_a] = eig(AAt);
    A = U_a * sqrt(max(D_a,0));
    sizep = size(Bp,1);
    R = [A Bp * U_22'; zeros(sizeG-3, 3) eye(sizeG-3)];
    blk_proj = [eye(3) zeros(3,sizeG-3); zeros(sizeG-3,3) U_22 * U_22'];
    G_vp = blk_proj' * G_v * blk_proj;
%    assert(false);
    
    for i=1:num_x
        basis = randn(d,3);
        basis = basis - Proj_half * (Proj_half' * basis);
        [basis, ~] = qr(basis, 0);
        x0_p = basis(:,1);
        x0_m = basis(:,2);
        w0 = basis(:,3);

        T_0 = [x0_p x0_m w0 mu (diag(c) \ M)'];
        G_0 = T_0' * T_0;
        
        %G_0
%        G_0 = G_0(sizeG:-1:1,sizeG:-1:1);
%        G_v = G_v(sizeG:-1:1,sizeG:-1:1);
%        R_0 = chol(G_0 + 1e-6 * eye(sizeG));
%        R_v = chol(G_v + 1e-6 * eye(sizeG));
%        R = R_0\R_v;
%        R = R(sizeG:-1:1,sizeG:-1:1);
%        G_0 = G_0(sizeG:-1:1,sizeG:-1:1);
%        G_v = G_v(sizeG:-1:1,sizeG:-1:1);
        
        T = [x0_p x0_m w0] * A' + [mu (diag(c) \ M)'] * U_22 * Bp';
        
        %G_v
        
        %(T' * T)
        T_full = [T mu (diag(c) \ M)'];
        err = norm(G_vp - (T_full'*T_full), 'inf');
        %err
        assert(err < 1e-0);
        assert(isreal(T));
        
        X_p(:,i) = T(:,1);
        X_m(:,i) = T(:,2);
        W(:,i) = T(:,3);
    end
end

function [U,D] = svd_lr(A, tol)
    [U,D,~] = svd(A, 'econ');
    active = diag(D) > tol*max(D(:));
    U = U(:,active);
    D = D(active,active);
end

function [U,D] = eig_lr(A, tol)
    [U,D] = eig((A+A')/2);
    active = diag(D)>tol*max(D(:));
    U = U(:, active);
    D = D(active,active);
end

function Ap = psd_proj(A)
    [U,D] = eig(A);
    D(D<1e-5) = 0;
    Ap = U*D*U';
    Ap = (Ap+Ap')/2;
end