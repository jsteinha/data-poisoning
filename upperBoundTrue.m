% G, Constraint are yalmip data for debugging
function [G, Constraint, val, X_eps, probs_eps] = upperBoundTrue(X_train, y_train, theta, bias, probs, mus, epsilon, r_slab, r_sphere, randomize, solver)
    % we don't have a good way of splitting u pthe probabilities, so let's
    % just do it randomly
    if randomize
        probs_eps = gamrnd([probs(1) probs(1) probs(2) probs(2)], 1);
        probs_eps = epsilon * probs_eps / sum(probs_eps);
    else
        % this heuristic choice also works well
        probs_eps = epsilon * [probs(1) 0 probs(2) 0];
    end

    % who are the relevant players?
    % x_a^+, x_b^+, x_a^-, x_b^-
    % mu^+, mu^-, theta
    G_o = sdpvar(4,4);
    G_s = sdpvar(4,3);
    G_m = [mus theta]' * [mus theta];
    G = [G_o G_s; G_s' G_m];
    Constraint = [G >= 0];
    e_ap = [1;0;0;0;0;0;0]; % x_a^+; this one is a support vector
    e_bp = [0;1;0;0;0;0;0]; % x_b^+; this one is not a support vector
    e_am = [0;0;1;0;0;0;0]; % x_a^-
    e_bm = [0;0;0;1;0;0;0]; % x_b^-
    e_up = [0;0;0;0;1;0;0]; % mu^+
    e_um = [0;0;0;0;0;1;0]; % mu^-
    e_th = [0;0;0;0;0;0;1]; % theta
    mu_pp = (probs(1) * e_up + probs_eps(1) * e_ap + probs_eps(2) * e_bp) / (probs(1) + probs_eps(1) + probs_eps(2)); %mu_poisoned^+
    mu_mp = (probs(2) * e_um + probs_eps(3) * e_am + probs_eps(4) * e_bm) / (probs(2) + probs_eps(3) + probs_eps(4)); %mu_poisoned^-
    
    % add inner product constraint
    Constraint = [Constraint;
                  1 - (e_ap' * G * e_th + bias) >= 0; % i.e., 1 - <x_a^+, theta> >= 0
                  1 - (e_bp' * G * e_th + bias) <= 0;
                  1 + (e_am' * G * e_th + bias) >= 0;
                  1 + (e_bm' * G * e_th + bias) <= 0];
    
    % add sphere constraints
    Constraint = [Constraint;
                  (e_ap - mu_pp)' * G * (e_ap - mu_pp) <= r_sphere(1)^2; % i.e., <x_a^+ - mu_poisoned^+, x_a^+ - mu_poisoned^+ > <= r_sphere^2
                  (e_bp - mu_pp)' * G * (e_bp - mu_pp) <= r_sphere(1)^2;
                  (e_am - mu_mp)' * G * (e_am - mu_mp) <= r_sphere(2)^2;
                  (e_bm - mu_mp)' * G * (e_bm - mu_mp) <= r_sphere(2)^2];
    
    % add slab constraints
    Constraint = [Constraint;
                  -r_slab(1) <= (e_ap - mu_pp)' * G * (mu_pp - mu_mp) <= r_slab(1); % i.e., -r_slab <= <x_a^+ - mu_poisoned^+, mu_poisoned^+ - mu_poisoned^_ > <= r_slab
                  -r_slab(1) <= (e_bp - mu_pp)' * G * (mu_pp - mu_mp) <= r_slab(1);
                  -r_slab(2) <= (e_am - mu_mp)' * G * (mu_pp - mu_mp) <= r_slab(2);
                  -r_slab(2) <= (e_bm - mu_mp)' * G * (mu_pp - mu_mp) <= r_slab(2)];

              
    Objective = probs_eps(1) * (1 - (e_ap' * G * e_th + bias)) + probs_eps(3) * (1 + (e_am' * G * e_th + bias)); % loss on the support vectors x_a^+ and x_a^-
    
    opts = sdpsettings('verbose', 0, 'showprogress', 0, 'solver', solver, 'cachesolvers', 1);
    optimize(Constraint, -Objective, opts);
    val = double(Objective);
    fprintf(1, 'value = %.4f \t (eps = [%.3f %.3f %.3f %.3f])\n', val, probs_eps(1), probs_eps(2), probs_eps(3), probs_eps(4));
    %[~, L0] = nabla_Loss(X_train, y_train, theta);
    %fprintf(1, 'upper bound: %.4f (all) | %.4f (L0) | %.4f (val)\n', L0 + val, L0, val);
    
    if nargout > 3
        X_eps = extractVecs(double(G), G_m, [mus theta]);

    %    G_approx = X_eps' * X_eps;

        X_eps = X_eps(:,1:4);
    end
end
