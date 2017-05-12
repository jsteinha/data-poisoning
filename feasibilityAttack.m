function feasibilityAttack(name, epsilon, eta, lambda, quantile)
    fprintf(1, 'generating slab attack (poisoned means)\n');
    fprintf(1, 'parameters settings:\n');
    fprintf(1, '\tepsilon = %.3f | eta = %.4f | lambda = %.3f | quantile = %.3f\n', epsilon, eta, lambda, quantile);

    load(sprintf('%s/%s_data.mat', name, name));
    %epsilon = input('enter value of epsilon (default 0.1): ');
    %quantile = input('enter quantile (default 0.6): ');
    %opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'gurobi', 'gurobi.TimeLimit', 3);
    [N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test, quantile);
    %epsilon_tape = [0.06 0.08 0.10 0.12 0.15 0.20 0.30 0.50 0.60 0.80];
    %eta = input('enter value of eta (default 0.10): '); %0.05;
    %lambda = input('enter value of lambda: ');
    z = zeros(d,1);
    theta = zeros(d,1);
    MAX_ITER = round(eta * N_train);
    X_pert = zeros(MAX_ITER, d);
    y_pert = zeros(MAX_ITER, 1);
    %
    % initial step
    [g_c, L_c] = nabla_Loss(X_train, y_train, theta);
    z = z - g_c;
    theta = theta - eta * g_c;
    % main loop
    opts = sdpsettings('verbose', 0, 'showprogress', 0, 'solver', 'gurobi', 'cachesolvers', 1);
    Rcum = 0.5 * eta * norm(g_c,2)^2;
    for iter = 1:MAX_ITER
        fprintf(1, '====== STARTING ITERATION %d ======\n', iter);
        vals = zeros(1,2);
        xs = zeros(d,2);
        for j=1:2
            s = sdpvar(d,1);
            t = sdpvar(d,1);
            x = s+t;
            ip = sdpvar(1);
            dmu_sq = (mus(:,1) - mus(:,2)).^2;
            Constraint = [ip == (x - mus(:,j))' * (mus(:,1) - mus(:,2));
                          sum(s+3/4) + norm(t-1/2, 2)^2  - 2 * x' * mus(:,j) + norm(mus(:,j), 2)^2 <= r_sphere(j)^2; 
                          ip^2 + 3 * x' * dmu_sq <= r_slab(j)^2;
                          %(x - mus(:,j))' * (mus(:,1) - mus(:,2)) >= -r_slab(j);
                          x >= 0; s <= -1/2];
            Objective = 1 - (3-2*j) * theta' * x;
            optimize(Constraint, -Objective, opts);
            vals(j) = double(Objective);
            xs(:,j) = double(x);
            xr = randRound(xs(:,j));
            fprintf(1, '\tfeasibility checks (y=%d): %.4f %.4f\n', ...
                3-2*j, norm(xr - mus(:,j),2)^2 / r_sphere(j)^2, ...
                ((xr - mus(:,j))' * (mus(:,1) - mus(:,2)))^2 / r_slab(j)^2);
        end
        fprintf(1, '\tvals: %.4f %.4f\n', vals(1), vals(2));
        if vals(1) > vals(2)
            j_max = 1;
            %X_pert(iter,:) = xs(:,1);
            y_pert(iter) = 1;
        else
            j_max = 2;
            %X_pert(iter,:) = xs(:,2);
            y_pert(iter) = -1;
        end
        [g_c, L_c] = nabla_Loss(X_train, y_train, theta);
        g_p = zeros(d,1);
        L_p = 0;
        NUM_SAMPLES = 100;
        for s = 1:NUM_SAMPLES
            xr = randRound(xs(:,j_max));
            [g_p_s, L_p_s] = nabla_Loss(xr', y_pert(iter), theta);
            g_p = g_p + g_p_s / NUM_SAMPLES;
            L_p = L_p + L_p_s / NUM_SAMPLES; 
        end
        X_pert(iter,:) = xr;
        fprintf(1, 'loss: %.4f (clean) | %.4f (poisoned) | %.4f (all)\n', L_c, L_p, L_c + epsilon * L_p);
        g = g_c + epsilon * g_p + 0.1 * theta;
        z = z - g;
        theta = z / (1/eta + iter * lambda);
        Rcum = Rcum + 0.5 * norm(g,2)^2 / (1/eta + iter * lambda);
        fprintf(1, '\nAVERAGE REGRET after %d iterations: %.4f + %.4f |theta|_2^2\n\n', iter, Rcum / iter, 0.5 / (eta * iter));
    end
    save(sprintf('%s/attacks/%s_attack_eps%02d_integer', name, name, round(100*epsilon)), 'X_train', 'X_pert', 'X_test', 'y_train', 'y_pert', 'y_test');
end