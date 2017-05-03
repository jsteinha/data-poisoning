%load('imdb.mat');
%load('xb_attack.mat');
N_pert = round(0.20 * N_train);
N_attack = mnrnd(N_pert, p_attack);
y_pert = [];
for j=1:k
  y_pert = [y_pert; j*ones(N_attack(j), 1)];
end
X_pert = sparse(N_pert,d);
B = 50;
for i=1:B:N_pert
    i
    i2 = min(i+B-1,N_pert);
    B2 = i2-i+1;
    %A_pert_cur = max(0, ones(B2,1) * mu_all' + y_pert(i:i2) * xb');
    X_pert_init = xb_full(:, y_pert(i:i2))';
    %X_pert_cur = max(0, X_pert_init);
    %X_pert_round = round(X_pert_cur);
    %X_pert_rem = X_pert_cur - X_pert_round;
    %X_pert_rem_sparse = 2 * sign(X_pert_rem) .* (rand(B2,d) < 0.5 * abs(X_pert_rem));
    %X_pert(i:i2,:) = X_pert_round + X_pert_rem_sparse;
    X_pert(i:i2,:) = X_pert_init;
end
clear X_pert_init X_pert_cur X_pert_round X_pert_rem X_pert_rem_sparse;
%% make sure data is actually poisoned
%[lossPertRev, accPertRev] = test(X_pert, y_pert, N_pert, d, theta);
%% train SVM
[lossPert, accPert, thetaPert] = trainMulticlass([X_train;X_pert], [y_train;y_pert], k, 0.005, 1e-4, N_train+N_pert, d, 99999);
[lossPertBad, accPertBad] = testMulticlass(X_pert, y_pert, k, N_pert, d, thetaPert, 99999);
[lossPertTest, accPertTest] = testMulticlass(X_test, y_test, k, N_test, d, thetaPert, 99999);
