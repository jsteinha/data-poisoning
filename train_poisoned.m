%load('imdb.mat');
%load('xb_attack.mat');
N_pert = round(0.10 * N);
y_pert = sign(randn(N_pert,1));
A_pert = sparse(N_pert,d);
B = 50;
for i=1:B:N_pert
    i
    i2 = min(i+B-1,N_pert);
    B2 = i2-i+1;
    A_pert_cur = max(0, ones(B2,1) * mu_all' + y_pert(i:i2) * xb');
    A_pert_round = round(A_pert_cur);
    A_pert_rem = A_pert_cur - A_pert_round;
    A_pert_rem_sparse = sign(A_pert_rem) .* (rand(B2,d) < abs(A_pert_rem));
    A_pert(i:i2,:) = A_pert_round + A_pert_rem_sparse;
end
clear A_pert_cur A_pert_round A_pert_rem A_pert_rem_sparse;
%% make sure data is actually poisoned
%[lossPertRev, accPertRev] = test(A_pert, y_pert, N_pert, d, theta);
%% train SVM
[lossPert, accPert, thetaPert] = train([A;A_pert], [yt;y_pert], 0.005, 1e-4, N+N_pert, d, 99999);
[lossPertBad, accPertBad] = test(A_pert, y_pert, N_pert, d, thetaPert);
[lossPertTest, accPertTest] = test(A_test, yt_test, N, d, thetaPert);
