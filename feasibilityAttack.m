function feasibilityAttack(name, epsilon, eta, lambda, quantile, solver)
    fprintf(1, 'generating slab attack (poisoned means)\n');
    fprintf(1, 'parameters settings:\n');
    fprintf(1, '\tepsilon = %.3f | eta = %.4f | lambda = %.3f | quantile = %.3f\n', epsilon, eta, lambda, quantile);
    disp('MULTIPLYING LAMBDA BY 1+EPSILON');
    lambda = lambda * (1+epsilon);
    fprintf(1, 'NEW VALUE OF LAMBDA: %.3f\n', lambda);

    load(sprintf('%s/%s_data.mat', name, name));
    [N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test, quantile);

    % initialize variables
    z = zeros(d,1);
    z_bias = 0;
    theta = zeros(d,1);
    bias = 0;
    % currently run for same number of iterations as the number of poisoned
    % points (no burn-in); can change this if necessary
    MAX_ITER = round(epsilon * N_train);
    X_pert = zeros(MAX_ITER, d);
    y_pert = zeros(MAX_ITER, 1);
    metadata = cell(MAX_ITER,1);
    
    % initial step
    [g_c, L_c, dbias_c] = nabla_Loss(X_train, y_train, theta, bias);
    z = z - g_c;
    z_bias = z_bias - dbias_c;
    theta = theta - eta * g_c;
    bias = bias - eta * dbias_c;
    
    % main loop
    opts = sdpsettings('verbose', 0, 'showprogress', 0, 'solver', solver, 'cachesolvers', 1);
    Rcum = 0.5 * eta * (norm(g_c,2)^2 + dbias_c^2);
    for iter = 1:MAX_ITER
        fprintf(1, '====== STARTING ITERATION %d ======\n', iter);
        vals = zeros(1,2);
        xs = zeros(d,2);
        for j=1:2
            % solve a QP to find a non-integer solution
            disp('solving QP'); tic;
            s = sdpvar(d,1);
            t = sdpvar(d,1);
            x = s+t;
            ip = sdpvar(1);
            dmu_sq = (mus(:,1) - mus(:,2)).^2;
            Constraint = [ip == (x - mus(:,j))' * (mus(:,1) - mus(:,2));
                          sum(s+3/4) + norm(t-1/2, 2)^2  - 2 * x' * mus(:,j) + norm(mus(:,j), 2)^2 <= r_sphere(j)^2; 
                          ip^2 + 2 * x' * dmu_sq <= r_slab(j)^2;
                          x >= 0; s <= -1/2];
            Objective = 1 - (3-2*j) * (theta' * x + bias);
            optimize(Constraint, -Objective, opts);
            x0 = double(x);
            val0 = double(Objective);
            toc;
            
            % do randomized rounding; take S samples, choose the best
            disp('rounding'); tic;
            S = 50;
            num_feas = 0;
            val_b = -inf;
            x_b = randRound(x0); % in case we don't get anything feasible
            for s=1:S
                x_c = randRound(x0);
                val_c = 1 - (3-2*j) * (theta' * x_c + bias);
                feas1_c = norm(x_c - mus(:,j),2)^2 / r_sphere(j)^2;
                feas2_c = ((x_c - mus(:,j))' * (mus(:,1) - mus(:,2)))^2 / r_slab(j)^2;
                if feas1_c <= 1 && feas2_c <= 1
                    num_feas = num_feas + 1;
                    if val_c > val_b
                        val_b = val_c;
                        x_b = x_c;
                    end
                end
            end
            toc;

            % save best value and print some statistics
            vals(j) = val_b;
            xs(:,j) = x_b;
            fprintf(1, '\tfeasible fraction: %.3f\n', num_feas / S);
            fprintf(1, '\tfeasibility checks (y=%d): %.4f %.4f\n', ...
                3-2*j, norm(x_b - mus(:,j),2)^2 / r_sphere(j)^2, ...
                ((x_b - mus(:,j))' * (mus(:,1) - mus(:,2)))^2 / r_slab(j)^2);
        end
        
        % take the value of y that leads to a larger loss
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
        [g_c, L_c, dbias_c, acc_c] = nabla_Loss(X_train, y_train, theta, bias);
        [g_p, L_p, dbias_p, acc_p] = nabla_Loss(xs(:,j_max)', y_pert(iter), theta, bias);
        X_pert(iter,:) = xs(:,j_max);
        
        % print the loss and some other stats
        fprintf(1, 'loss: %.4f (clean) | %.4f (poisoned) | %.4f (all)\n', L_c, L_p, L_c + epsilon * L_p);
        fprintf(1, ' acc: %.4f (clean) | %.4f (poisoned)\n', acc_c, acc_p);
        fprintf(1, 'norm of params: %.4f | bias: %.4f\n', norm(theta,2), bias);
        metadata{iter} = struct('L_c', L_c, 'L_p', L_p, 'acc_c', acc_c, 'acc_p', acc_p, 'norm_theta', norm(theta,2), 'bias', bias);
        
        % do gradient update
        g = g_c + epsilon * g_p;
        dbias = dbias_c + epsilon * dbias_p;
        z = z - g;
        z_bias = z_bias - dbias;
        theta = z / (1/eta + iter * lambda);
        bias = z_bias / (1/eta + iter * lambda);
        
        % output bound on regret
        Rcum = Rcum + 0.5 * (norm(g,2)^2 + dbias^2) / (1/eta + iter * lambda);
        fprintf(1, '\nAVERAGE REGRET after %d iterations: %.4f + %.4f |theta|_2^2\n\n', iter, Rcum / iter, 0.5 / (eta * iter));
    end
    metadata_final = metadata{MAX_ITER};
    save(sprintf('%s/attacks/%s_attack_eps%02d_integer', name, name, round(100*epsilon)), 'X_train', 'X_pert', 'X_test', 'y_train', 'y_pert', 'y_test', ...
        'theta', 'bias', 'Rcum', ...
        'epsilon', 'eta', 'lambda', 'metadata', 'metadata_final');
end
