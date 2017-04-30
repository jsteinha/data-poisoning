%load('imdb.mat');
%load('xb_attack.mat');
N_pert = round(0.10 * N_train);
y_pert = sign(randn(N_pert,1));
X_pert = sparse(N_pert,d);
B = 50;
for i=1:B:N_pert
    i
    i2 = min(i+B-1,N_pert);
    B2 = i2-i+1;
    %A_pert_cur = max(0, ones(B2,1) * mu_all' + y_pert(i:i2) * xb');
    X_pert_cur = max(0, (y_pert(i:i2)==1) * xb_plus' + (y_pert(i:i2)==-1) * xb_minus');
    X_pert_round = round(X_pert_cur);
    X_pert_rem = X_pert_cur - X_pert_round;
    X_pert_rem_sparse = 2 * sign(X_pert_rem) .* (rand(B2,d) < 0.5 * abs(X_pert_rem));
    X_pert(i:i2,:) = X_pert_round + X_pert_rem_sparse;
end
clear X_pert_cur X_pert_round X_pert_rem X_pert_rem_sparse;
%% make sure data is actually poisoned
%[lossPertRev, accPertRev] = test(X_pert, y_pert, N_pert, d, theta);
%% train SVM
[lossPert, accPert, thetaPert] = train([X_train;X_pert], [y_train;y_pert], 0.005, 1e-4, N_train+N_pert, d, 1000);
[lossPertBad, accPertBad] = test(X_pert, y_pert, N_pert, d, thetaPert, 2500);
[lossPertTest, accPertTest] = test(X_test, y_test, N_test, d, thetaPert, 2500);
