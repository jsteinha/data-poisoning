clear;
name = 'imdb';
load(sprintf('%s/%s_data.mat', name, name));
opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'sedumi');
[N_train, N_test, d, mus, probs, r_sphere, r_slab, r_ones] = processDataLight(X_train, y_train, X_test, y_test);
epsilon_tape = [0.06 0.08 0.10 0.12 0.15 0.20 0.30 0.50 0.60 0.80];
nmlz = @(x) x/norm(x,2);
%%
for epsilon = epsilon_tape
    [xb_plus, xb_minus, w_nom] = generateAttack(epsilon, mu, probs, r_sphere, r_slab, [], [], [], [], opts, dataOpts);
    [thetaPert, accGood, accBad, X_pert, y_pert] = trainPoisonedFnc(xb_plus, xb_minus, X_train, y_train, epsilon, p_plus, dataOpts);
    accGood = flip(accGood); accBad = flip(accBad);
    
    accGoods = [accGoods accGood];
    accBads = [accBads accBad];
    ips = [ips [xb_plus' * thetaPert; xb_minus' * thetaPert]];
    [~,L0] = nabla_Loss(X_train, y_train, thetaPert, 0);
    losses = [losses L0];
    L0full = L0 + epsilon * p_plus * max(1 - xb_plus' * thetaPert, 0) ...
                + epsilon * p_minus * max(1 + xb_minus' * thetaPert, 0);
    lossesFull = [lossesFull L0full];
    [~,Ltest] = nabla_Loss(X_test, y_test, thetaPert, 0);
    lossesTest = [lossesTest Ltest];
    
    fprintf(1, 'STATS FOR epsilon = %.2f\n', epsilon);
    fprintf(1, '\tgood accuracy: %.3f %.3f\n', accGood(1), accGood(2));
    fprintf(1, '\t bad accuracy: %.3f %.3f\n', accBad(1), accBad(2));
    fprintf(1, '\tloss: %.3f (clean) | %.3f (full) | %.3f (test)\n', L0, L0full, Ltest);
    stats = struct('accGood', accGood, 'accBad', accBad, 'lossTrainClean', L0, 'lossTrainFull', L0full, 'lossTest', Ltest);
    theta = thetaPert;
    
    save(fprintf(1, '%s/%s_mean_attack_eps%02d', name, name, epsilon), 'X_pert', 'y_pert', 'X_train', 'y_train', 'X_test', 'y_test', 'stats', 'theta');
end