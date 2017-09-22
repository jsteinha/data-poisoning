function poolAttack(name, epsilon, eta, lambda)
    fprintf(1, 'generating slab attack (poisoned means)\n');
    fprintf(1, 'parameters settings:\n');
    fprintf(1, '\tepsilon = %.3f | eta = %.4f | lambda = %.3f\n', epsilon, eta, lambda);
    disp('MULTIPLYING LAMBDA BY 1+EPSILON');
    lambda = lambda * (1+epsilon);
    fprintf(1, 'NEW VALUE OF LAMBDA: %.3f\n', lambda);

    load(sprintf('%s/%s_data_pool1.mat', name, name));
    [N_train, N_test, d] = processDataLight(X_train, y_train, X_test, y_test, 0.99);

    % initialize variables
    z = zeros(d,1);
    z_bias = 0;
    theta = zeros(d,1);
    bias = 0;
    % currently run for same number of iterations as the number of poisoned
    % points (no burn-in); can change this if necessary
    batch_size = round(epsilon * N_train);
    MAX_ITER = max(2000, batch_size); %round(epsilon * N_train));
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
    %opts = sdpsettings('verbose', 0, 'showprogress', 0, 'solver', solver, 'cachesolvers', 1);
    Rcum = 0.5 * eta * (norm(g_c,2)^2 + dbias_c^2);
    for iter = 1:MAX_ITER
        fprintf(1, '====== STARTING ITERATION %d ======\n', iter);
        scores = 1 - y_pool .* (X_pool * theta + bias);
        [v_max,i_max] = max(scores);
        y_pert(iter) = y_pool(i_max);
        X_pert(iter,:) = X_pool(i_max,:);
        
        [g_c, L_c, dbias_c, acc_c] = nabla_Loss(X_train, y_train, theta, bias);
        [g_p, L_p, dbias_p, acc_p] = nabla_Loss(X_pert(iter,:), y_pert(iter), theta, bias);
        
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
        theta = z / (max(1/eta, iter * lambda));
        bias = z_bias / (max(1/eta, iter * lambda));
        
        % output bound on regret
        Rcum = Rcum + 0.5 * (norm(g,2)^2 + dbias^2) / (max(1/eta, iter * lambda));
        fprintf(1, '\nAVERAGE REGRET after %d iterations: %.4f + %.4f |theta|_2^2\n\n', iter, Rcum / iter, (0.5 / iter) * max(1/eta - iter * lambda, 0));
        
        if mod(iter, batch_size) == 0
            
    end
    metadata_final = metadata{MAX_ITER};
    save(sprintf('%s/attacks/%s_attack_eps%02d_integer', name, name, round(100*epsilon)), 'X_train', 'X_pert', 'X_test', 'y_train', 'y_pert', 'y_test', ...
        'theta', 'bias', 'Rcum', ...
        'epsilon', 'eta', 'lambda', 'metadata', 'metadata_final');
end
