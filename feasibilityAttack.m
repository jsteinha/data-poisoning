name = 'enron';
load(sprintf('%s/%s_data.mat', name, name));
epsilon = 0.3;
%opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'gurobi', 'gurobi.TimeLimit', 3);
[N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test);
%epsilon_tape = [0.06 0.08 0.10 0.12 0.15 0.20 0.30 0.50 0.60 0.80];
eta = input('enter value of eta (default 0.05): '); %0.05;
delta = input('enter value of delta (default 1.0): '); %1.0;
theta = zeros(d,1);
theta2 = zeros(d,1);
MAX_ITER = 1000;
X_pert = zeros(MAX_ITER, d);
y_pert = zeros(MAX_ITER, 1);
%
% initial step
[g_c, L_c] = nabla_Loss(X_train, y_train, theta);
theta2 = theta2 + g_c .* g_c;
theta = theta - eta * g_c ./ (delta + sqrt(theta2));
% main loop
opts = sdpsettings('verbose', 0, 'showprogress', 0, 'solver', 'gurobi', 'gurobi.TimeLimit', 3);
for iter = 1:MAX_ITER
    fprintf(1, '====== STARTING ITERATION %d ======\n', iter);
    vals = zeros(1,2);
    xs = zeros(d,2);
    for j=1:2
        s = sdpvar(d,1);
        t = sdpvar(d,1);
        x = s+t;
        Constraint = [sum(s+3/4) + norm(t-1/2, 2)^2  - 2 * x' * mus(:,j) + norm(mus(:,j), 2)^2 <= r_sphere(j)^2; 
                      (x - mus(:,j))' * (mus(:,1) - mus(:,2)) <= r_slab(j);
                      (x - mus(:,j))' * (mus(:,1) - mus(:,2)) >= -r_slab(j);
                      x >= 0; s <= -1/2];
        Objective = 1 - (3-2*j) * theta' * x;
        optimize(Constraint, -Objective, opts);
        vals(j) = double(Objective);
        xs(:,j) = double(x);
        xr = randRound(xs(:,j));
        fprintf(1, '\tfeasibility checks (y=%d): %.4f %.4f\n', ...
            3-2*j, norm(xr - mus(:,j),2)^2 / r_sphere(j)^2, ...
            abs((xr - mus(:,j))' * (mus(:,1) - mus(:,2))) / r_slab(j));
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
    eta_t = eta; % / sqrt(1 + 0.1 * iter);
    theta2 = theta2 + g .* g;
    theta = theta - eta_t * g ./ (delta + sqrt(theta2));
end
