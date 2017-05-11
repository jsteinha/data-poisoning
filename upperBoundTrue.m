function upperBoundTrue(X_train, y_train, theta, probs, mus, epsilon, r_slab, r_sphere)
    probs_eps = gamrnd([probs(1) probs(1) probs(2) probs(2)], 1);
    probs_eps = epsilon * probs_eps / sum(probs_eps);
    % who are the players?
    % x_a^+, x_b^+, x_a^-, x_b^-
    % mu^+, mu^-, theta
    G_o = sdpvar(4,4);
    G_s = sdpvar(4,3);
    G_m = [mus theta]' * [mus theta];
    G = [G_o G_s; G_s' G_m];
    Constraint = [G >= 0];
    e_ap = [1;0;0;0;0;0;0];
    e_bp = [0;1;0;0;0;0;0];
    e_am = [0;0;1;0;0;0;0];
    e_bm = [0;0;0;1;0;0;0];
    e_up = [0;0;0;0;1;0;0];
    e_um = [0;0;0;0;0;1;0];
    e_th = [0;0;0;0;0;0;1];
    mu_pp = (probs(1) * e_up + probs_eps(1) * e_ap + probs_eps(2) * e_bp) / (probs(1) + probs_eps(1) + probs_eps(2));
    mu_mp = (probs(2) * e_um + probs_eps(3) * e_am + probs_eps(4) * e_bm) / (probs(2) + probs_eps(3) + probs_eps(4));
    
    % add inner product constraint
    Constraint = [Constraint;
                  1 - e_ap' * G * e_th >= 0;
                  1 - e_bp' * G * e_th <= 0;
                  1 + e_am' * G * e_th >= 0;
                  1 + e_bm' * G * e_th <= 0];
    
    % add sphere constraints
    Constraint = [Constraint;
                  (e_ap - mu_pp)' * G * (e_ap - mu_pp) <= r_sphere(1)^2;
                  (e_bp - mu_pp)' * G * (e_bp - mu_pp) <= r_sphere(1)^2;
                  (e_am - mu_mp)' * G * (e_ap - mu_mp) <= r_sphere(1)^2;
                  (e_bm - mu_mp)' * G * (e_bp - mu_mp) <= r_sphere(1)^2];
    
    % add slab constraints
    Constraint = [Constraint;
                  -r_slab(1) <= (e_ap - mu_pp)' * G * (mu_pp - mu_mp) <= r_slab(1);
                  -r_slab(1) <= (e_bp - mu_pp)' * G * (mu_pp - mu_mp) <= r_slab(1);
                  -r_slab(2) <= (e_am - mu_mp)' * G * (mu_pp - mu_mp) <= r_slab(2);
                  -r_slab(2) <= (e_bm - mu_mp)' * G * (mu_pp - mu_mp) <= r_slab(2)];

              
    Objective = probs_eps(1) * (1 - e_ap' * G * e_th) + probs_eps(3) * (1 + e_am' * G * e_th);
    
    opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'gurobi');
    optimize(Constraint, -Objective, opts);
    val = double(Objective);
    fprintf(1, 'value = %.4f \t (eps = [%.3f %.3f %.3f %.3f])\n', val, probs_eps(1), probs_eps(2), probs_eps(3), probs_eps(4));
    [~, L0] = nabla_Loss(X_train, y_train, theta);
    fprintf(1, 'upper bound: %.4f (all) | %.4f (L0) | %.4f (val)\n', L0 + val, L0, val);
end