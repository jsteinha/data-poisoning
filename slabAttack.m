function [bestLower, bestUpper, lower_bounds, upper_bounds] = slabAttack(name, epsilon, eta, lambda, quantile)
    fprintf(1, 'generating slab attack (poisoned means)\n');
    fprintf(1, 'parameters settings:\n');
    fprintf(1, '\tepsilon = %.3f | eta = %.4f | lambda = %.3f | quantile = %.3f\n', epsilon, eta, lambda, quantile);
    load(sprintf('%s/%s_data.mat', name, name));
    [N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test, quantile);
    z = zeros(d,1);
    theta = zeros(d,1);
    MAX_ITER = round(epsilon * N_train);
    X_pert = zeros(MAX_ITER, d);
    y_pert = zeros(MAX_ITER, 1);
    losses = zeros(MAX_ITER, 5);

    % initial step
    [g_c, L_c] = nabla_Loss(X_train, y_train, theta);
    z = z - g_c;
    theta = theta - eta * g_c;
    Rcum = 0.5 * eta * norm(g_c,2)^2;

    upper_bounds = [];
    lower_bounds = [];
    NUM_K = 5;
    N_pert = round(epsilon * N_train);
    bestX = zeros(NUM_K, N_pert, d);
    bestY = zeros(NUM_K, N_pert);
    bestV = -inf * ones(NUM_K, 1);
    for iter = 1:MAX_ITER
        fprintf(1, '====== STARTING ITERATION %d ======\n', iter);
        xs = zeros(d,2);
        [~,~,val,X_eps,probs_eps] = upperBoundTrue(X_train, y_train, theta, probs, mus, epsilon, r_slab, r_sphere, 0);
        xs(:,1) = X_eps(:,1);
        xs(:,2) = X_eps(:,3);
        fprintf(1, '\tval: %.4f\n', val);
        [g_c, L_c] = nabla_Loss(X_train, y_train, theta);
        [g_pp, L_pp] = nabla_Loss(xs(:,1)', 1, theta);
        [g_pm, L_pm] = nabla_Loss(xs(:,2)', -1, theta);
        fprintf(1, 'loss: %.4f (clean) | %.4f, %.4f (poisoned) | %.4f (norm_sq) | %.4f (all)\n', L_c, L_pp, L_pm, norm(theta,2)^2, L_c + probs_eps(1) * L_pp + probs_eps(3) * L_pm + 0.5 * lambda * norm(theta,2)^2);
        losses(iter, :) = [L_c, L_pp, L_pm, norm(theta,2)^2, L_c + probs_eps(1) * L_pp + probs_eps(3) * L_pm + 0.5 * lambda * norm(theta,2)^2];
        g = g_c + probs_eps(1) * g_pp + probs_eps(3) * g_pm;
        z = z - g;
        theta = z / (1/eta + iter * lambda);
        Rcum = Rcum + 0.5 * norm(g,2)^2 / (1/eta + iter * lambda);
        fprintf(1, '\nAVERAGE REGRET after %d iterations: %.4f + %.4f |theta|_2^2\n\n', iter, Rcum / iter, 0.5 / (eta * iter));

        if mod(iter, 10) == 0
            fprintf(1, 'Checking lower bound...\n');
            y_eps = [1 1 -1 -1];
            N_pert = round(epsilon * N_train);
            choices = mnrnd(1, probs_eps / sum(probs_eps), N_pert); %N_pert x 4
            X_pert_t = choices * X_eps';
            y_pert_t = choices * y_eps';
            N_tot = N_train + N_pert;
            [loss, acc, theta_pert] = train([X_train;X_pert_t], [y_train;y_pert_t], 0.05, 1.0, N_tot, d, 99999, 5, lambda/(1+epsilon), 0);
            loss = (1+epsilon) * loss;
            fprintf(1, '\n\t***********************\n');
            fprintf(1, '\t** LOWER BOUND: %.4f **\n', loss);
            fprintf(1, '\t*************************\n\n');
            lower_bounds = [lower_bounds; loss];
            if loss > bestV(NUM_K)
                bestV(NUM_K) = loss;
                bestX(NUM_K,:,:) = X_pert_t;
                bestY(NUM_K,:) = y_pert_t;
                k = NUM_K;
                while k > 1
                    if bestV(k) > bestV(k-1)
                        [bestV(k), bestV(k-1)] = deal(bestV(k-1), bestV(k));
                        [bestX(k,:,:), bestX(k-1,:,:)] = deal(bestX(k-1,:,:), bestX(k,:,:));
                        [bestY(k,:), bestY(k-1,:)] = deal(bestY(k-1,:), bestY(k,:));
                        k = k-1;
                    else
                        break;
                    end
                end
            end
        end
        if mod(iter, 50) == 0
            S = 200;
            fprintf(1, 'Generating true upper bound (%d trials)...\n', S);
            maxVal = -inf;
            for s=1:S
                [~,~,val] = upperBoundTrue(X_train, y_train, theta, probs, mus, epsilon, r_slab, r_sphere, 1);
                maxVal = max(val, maxVal);
            end
            fprintf(1, '\tupper bound: %.4f\n', maxVal);
            upper_bounds = [upper_bounds; maxVal];
        end
    end
    bestLower = max(lower_bounds);
    bestUpper = min(upper_bounds);
    fprintf(1, 'best lower bound: %.4f\n', bestLower);
    fprintf(1, 'best upper bound: %.4f\n', bestUpper);
    
    for k=1:NUM_K
        X_pert = squeeze(bestX(k,:,:));
        y_pert = bestY(k,:)';
        save(sprintf('%s/attacks/%s_attack_eps%02d_slab%d', name, name, round(100*epsilon), k), 'X_train', 'X_pert', 'X_test', 'y_train', 'y_pert', 'y_test');
    end
    
end