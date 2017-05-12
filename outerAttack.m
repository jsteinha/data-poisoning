clear all;
name = 'dogfish';
load(sprintf('%s/%s_data.mat', name, name));
epsilon = 0.3;
%opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'gurobi', 'gurobi.TimeLimit', 3);
[N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test);
%epsilon_tape = [0.06 0.08 0.10 0.12 0.15 0.20 0.30 0.50 0.60 0.80];
%eta = 0.1;
%delta = 1.0;
eta = 0.025;
z = zeros(d,1);
theta = zeros(d,1);
%theta2 = zeros(d,1);
MAX_ITER = round(epsilon * N_train);
X_pert = zeros(MAX_ITER, d);
y_pert = zeros(MAX_ITER, 1);
losses = zeros(MAX_ITER, 5);
%%
% initial step
[g_c, L_c] = nabla_Loss(X_train, y_train, theta);
%theta2 = theta2 + g_c .* g_c;
%theta = theta - eta * g_c ./ (delta + sqrt(theta2));
z = z - g_c;
theta = theta - eta * g_c;
% main loop
opts = sdpsettings('verbose', 0, 'showprogress', 0, 'solver', 'gurobi', 'gurobi.TimeLimit', 3);
lambda = 0.6; %0.05;
%%
Rcum = 0.5 * eta * norm(g_c,2)^2;
for iter = 1:MAX_ITER
    fprintf(1, '====== STARTING ITERATION %d ======\n', iter);
    vals = zeros(1,2);
    xs = zeros(d,2);
%    disp('solving optimization...');
%    tic;
    for j=1:2
%        disp('QR');
%        tic;
        [Q,~] = qr([mus theta], 0);
%        toc;
%        disp('projecting');
%        tic;
        d_proj = size(Q,2);
        P = zeros(0,d_proj);
        x_proj = sdpvar(d_proj,1);
        mus_proj = Q' * mus;
        theta_proj = Q' * theta;
%        toc;
%        disp('solve optimization');
%        tic;
        Constraint = [norm(x_proj - mus_proj(:,j), 2) <= r_sphere(j); 
                      (x_proj - mus_proj(:,j))' * (mus_proj(:,1) - mus_proj(:,2)) <= r_slab(j);
                      (x_proj - mus_proj(:,j))' * (mus_proj(:,1) - mus_proj(:,2)) >= -r_slab(j);
                      P * x_proj >= 0;];
        Objective = 1 - (3-2*j) * theta_proj' * x_proj;
        optimize(Constraint, -Objective, opts);
%        toc;
        vals(j) = double(Objective);
        xs(:,j) = double(Q*x_proj);
        xr = xs(:,j);
        fprintf(1, '\tfeasibility checks (y=%d): %.4f %.4f\n', ...
            3-2*j, norm(xr - mus(:,j),2)^2 / r_sphere(j)^2, ...
            abs((xr - mus(:,j))' * (mus(:,1) - mus(:,2))) / r_slab(j));
    end
%    toc;
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
 %   disp('taking gradient...');
 %   tic;
    [g_c, L_c] = nabla_Loss(X_train, y_train, theta);
    g_p = zeros(d,1);
    L_p = 0;
    NUM_SAMPLES = 1;
    for s = 1:NUM_SAMPLES
        xr = xs(:,j_max);
        [g_p_s, L_p_s] = nabla_Loss(xr', y_pert(iter), theta);
        g_p = g_p + g_p_s / NUM_SAMPLES;
        L_p = L_p + L_p_s / NUM_SAMPLES; 
    end
    X_pert(iter,:) = xr;
 %   toc;
    fprintf(1, 'loss: %.4f (clean) | %.4f (poisoned) | %.4f (norm_sq) | %.4f (all)\n', L_c, L_p, norm(theta,2)^2, L_c + epsilon * L_p + 0.5 * lambda * norm(theta,2)^2);
    losses(iter, :) = [L_c, L_p, norm(theta,2)^2, L_c + epsilon * L_p + 0.5 * lambda * norm(theta,2)^2];
    g = g_c + epsilon * g_p; % + lambda * theta;
    %theta2 = theta2 + g .* g;
    % initial: eta/delta, meaning reg is delta/eta
    % end up at delta/eta + lambda * iter
    % 
    %theta = theta - eta * g ./ (delta + eta * lambda * iter + sqrt(theta2));
    z = z - g;
    theta = z / (1/eta + iter * lambda);
    %theta = (theta - eta * g) / (1 + eta * lambda);
    Rcum = Rcum + 0.5 * norm(g,2)^2 / (1/eta + iter * lambda);
    fprintf(1, '\nAVERAGE REGRET after %d iterations: %.4f + %.4f |theta|_2^2\n\n', iter, Rcum / iter, 0.5 / (eta * iter));
end
%%
Rcum = 0.5 * eta * norm(g_c,2)^2;
lower_bounds = [];
for iter = 1:MAX_ITER
    fprintf(1, '====== STARTING ITERATION %d ======\n', iter);
    xs = zeros(d,2);
    [~,~,val,X_eps,probs_eps] = upperBoundTrue(X_train, y_train, theta, probs, mus, epsilon, r_slab, r_sphere);
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
        fprintf(1, '\t***********************\n\n');
        lower_bounds = [lower_bounds; loss];
    end
end
%%
[G, Constraint, X_eps, probs_eps] = upperBoundTrue(X_train, y_train, theta, probs, mus, epsilon, r_slab, r_sphere);
%%
y_eps = [1 1 -1 -1];
N_pert = round(epsilon * N_train);
choices = mnrnd(1, probs_eps / sum(probs_eps), N_pert); %N_pert x 4
% X_eps is d x 4
X_pert_t = choices * X_eps';
y_pert_t = choices * y_eps';
%%
eta = 0.05; delta = 1.0;
N_tot = N_train + N_pert;
[loss, acc, theta_pert] = train([X_train;X_pert_t], [y_train;y_pert_t], eta, delta, N_tot, d, 99999, 5, lambda/(1+epsilon), 0);
loss = (1+epsilon) * loss;
fprintf(1, 'lower bound: %.4f\n', loss);
%%
eta = 0.05; delta = 1.0; lambda = 0.6; epsilon = 0.1;
X_train2 = repmat(X_train, [1 1]);
y_train2 = repmat(y_train, [1 1]);
N_tot = 1 * N_train + length(y_pert);
[loss, acc, theta_pert] = train([X_train2;X_pert], [y_train2;y_pert], eta, delta, N_tot, d, 99999, 10, lambda);
%%
[~, L_c] = nabla_Loss(X_train, y_train, theta_pert);
[~, L_p] = nabla_Loss(X_pert, y_pert, theta_pert);
fprintf(1, 'loss: %.4f (clean) | %.4f (poisoned) | %.4f (norm_sq) | %.4f (all)\n', L_c, L_p, norm(theta_pert,2)^2, L_c + epsilon * L_p + 0.5 * lambda * norm(theta_pert,2)^2);
%%
load jacob_dogfish_wd-0.6_params.mat
w=w';
[~, L_c] = nabla_Loss(X_train, y_train, w);
[~, L_p] = nabla_Loss(X_pert, y_pert, w);
fprintf(1, 'loss: %.4f (clean) | %.4f (poisoned) | %.4f (norm_sq) | %.4f (all)\n', L_c, L_p, norm(w,2)^2, L_c + epsilon * L_p + 0.5 * lambda * norm(w,2)^2);