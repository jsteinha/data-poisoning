clear;
name = 'enron';
load(sprintf('%s/%s_data.mat', name, name));
opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'gurobi', 'gurobi.TimeLimit', 3);
[N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test);
epsilon_tape = [0.06 0.08 0.10 0.12 0.15 0.20 0.30 0.50 0.60 0.80];
%%
opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'sedumi');
x = sdpvar;
y = max(x - 0.5, 0)^2;
z = x + y;
Objective = z;
Constraint = [-2 <= x <= 2];
optimize(Constraint, Objective, opts);
%%
w = randn(d,1);
%%
%x = sdpvar(d,1);
eta = 0.05;
delta = 1.0;
theta = zeros(d,1);
theta2 = zeros(d,1);
MAX_ITER = 1000;
X_pert = zeros(MAX_ITER, d);
y_pert = zeros(MAX_ITER, 1);
epsilon = 0.1;
%%
% initial step
[g_c, L_c] = nabla_Loss(X_train, y_train, theta);
theta2 = theta2 + g .* g;
theta = theta - eta * g ./ (delta + sqrt(theta2));
% main loop
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
        opts = sdpsettings('verbose', 0, 'showprogress', 0, 'solver', 'gurobi', 'gurobi.TimeLimit', 3);
        optimize(Constraint, -Objective, opts);
        vals(j) = double(Objective);
        xs(:,j) = randRound(double(x));
        fprintf(1, '\tfeasibility checks (y=%d): %.4f %.4f\n', ...
            3-2*j, r_sphere(j)^2 / norm(xs(:,j) - mus(:,j),2)^2, ...
            abs((xs(:,j) - mus(:,j))' * (mus(:,1) - mus(:,2))) / r_slab(j));
    end
    fprintf(1, '\tvals: %.4f %.4f\n', vals(1), vals(2));
    if vals(1) > vals(2)
        X_pert(iter,:) = xs(:,1);
        y_pert(iter) = 1;
    else
        X_pert(iter,:) = xs(:,2);
        y_pert(iter) = -1;
    end
    [g_c, L_c] = nabla_Loss(X_train, y_train, theta);
    [g_p, L_p] = nabla_Loss(X_pert(iter,:), y_pert(iter), theta);
    fprintf(1, 'loss: %.4f (clean) | %.4f (poisoned) | %.4f (all)\n', L_c, L_p, L_c + epsilon * L_p);
    g = g_c + epsilon * g_p;
    eta_t = eta; % / sqrt(1 + 0.1 * iter);
    theta2 = theta2 + g .* g;
    theta = theta - eta_t * g ./ (delta + sqrt(theta2));
end
%%
x = sdpvar(d,1);
Constraint = [norm(x - mus(:,1),2) <= r_sphere(1); 
              (x - mus(:,1))' * (mus(:,1) - mus(:,2)) <= r_slab(1);
              (x - mus(:,1))' * (mus(:,1) - mus(:,2)) >= -r_slab(1);
              x >= 0];
Objective = w' * x;
%%
opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'gurobi');
optimize(Constraint, Objective, opts);
%%
accGoods = [];
accBads = [];
ips = [];
losses = [];
lossesFull = [];
lossesTest = [];
for epsilon = epsilon_tape
    [xb_plus, xb_minus, w_nom] = generateAttack(epsilon, mus, probs, r_sphere, r_slab, [], [], [], [], opts, dataOpts);
    [thetaPert, accGood, accBad, X_pert, y_pert] = trainPoisonedFnc(xb_plus, xb_minus, X_train, y_train, epsilon, probs(1), dataOpts);
    accGood = flip(accGood); accBad = flip(accBad);
    
    accGoods = [accGoods accGood];
    accBads = [accBads accBad];
    ips = [ips [xb_plus' * thetaPert; xb_minus' * thetaPert]];
    [~,L0] = nabla_Loss(X_train, y_train, thetaPert, 0);
    losses = [losses L0];
    L0full = L0 + epsilon * probs(1) * max(1 - xb_plus' * thetaPert, 0) ...
                + epsilon * probs(2) * max(1 + xb_minus' * thetaPert, 0);
    lossesFull = [lossesFull L0full];
    [~,Ltest] = nabla_Loss(X_test, y_test, thetaPert, 0);
    lossesTest = [lossesTest Ltest];
    
    fprintf(1, 'STATS FOR epsilon = %.2f\n', epsilon);
    fprintf(1, '\tgood accuracy: %.3f %.3f\n', accGood(1), accGood(2));
    fprintf(1, '\t bad accuracy: %.3f %.3f\n', accBad(1), accBad(2));
    fprintf(1, '\tinner products: %.3f %.3f\n', ips(1,end), ips(2,end));
    fprintf(1, '\tloss: %.3f (clean) | %.3f (full) | %.3f (test)\n', L0, L0full, Ltest);
    stats = struct('accGood', accGood, 'accBad', accBad, 'lossTrainClean', L0, 'lossTrainFull', L0full, 'lossTest', Ltest);
    theta = thetaPert;
    save(sprintf('%s/%s_mean_attack_eps%02d', name, name, round(100*epsilon)), 'X_pert', 'y_pert', 'X_train', 'y_train', 'X_test', 'y_test', 'stats', 'theta');
end