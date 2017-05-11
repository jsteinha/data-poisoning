clear all;
name = 'dogfish';
load(sprintf('%s/%s_data.mat', name, name));
epsilon = 0.1;
%opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'gurobi', 'gurobi.TimeLimit', 3);
[N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test);
%epsilon_tape = [0.06 0.08 0.10 0.12 0.15 0.20 0.30 0.50 0.60 0.80];
eta = 0.05;
delta = 1.0;
theta = zeros(d,1);
theta2 = zeros(d,1);
MAX_ITER = 1000; %3310;
X_pert = zeros(MAX_ITER, d);
y_pert = zeros(MAX_ITER, 1);
%
% initial step
[g_c, L_c] = nabla_Loss(X_train, y_train, theta);
theta2 = theta2 + g_c .* g_c;
theta = theta - eta * g_c ./ (delta + sqrt(theta2));
% main loop
opts = sdpsettings('verbose', 0, 'showprogress', 0, 'solver', 'gurobi', 'gurobi.TimeLimit', 3);
%%
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
    fprintf(1, 'loss: %.4f (clean) | %.4f (poisoned) | %.4f (all)\n', L_c, L_p, L_c + epsilon * L_p);
    g = g_c + epsilon * g_p;
    eta_t = eta; % / sqrt(1 + 0.1 * iter);
    theta2 = theta2 + g .* g;
    theta = theta - eta_t * g ./ (delta + sqrt(theta2));
end
%%
X_train2 = repmat(X_train, [4 1]);
y_train2 = repmat(y_train, [4 1]);
N_tot = 4 * N_train + length(y_pert);
[loss, acc, theta_pert] = train([X_train2;X_pert], [y_train2;y_pert], 3*eta, delta, N_tot, d, 99999, 5);
