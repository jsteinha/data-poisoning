function [thetaPert, accGood, accBad, X_pert, y_pert] = trainPoisonedFnc(xb_plus, xb_minus, X_train, y_train, epsilon, p_plus, enforce_pos)
    N_train = size(X_train, 1);
    d = size(X_train, 2);
    N_pert = round(epsilon * N_train);
    y_pert = 2*(rand(N_pert,1)<p_plus)-1;
    X_pert = sparse(N_pert,d);
    B = 100;
    for i=1:B:N_pert
        i
        i2 = min(i+B-1,N_pert);
        B2 = i2-i+1;
        %A_pert_cur = max(0, ones(B2,1) * mu_all' + y_pert(i:i2) * xb');
        X_pert_init = (y_pert(i:i2)==1) * xb_plus' + (y_pert(i:i2)==-1) * xb_minus';
        %X_pert_init = diag(y_pert(i:i2)==1) * X_plus(:,i:i2)' + diag(y_pert(i:i2)==-1) * X_minus(:,i:i2)';
        if ~enforce_pos
            X_pert(i:i2,:) = X_pert_init;
        else
            X_pert_cur = max(0, X_pert_init);
            X_pert_round = round(X_pert_cur);
            X_pert_rem = X_pert_cur - X_pert_round;
            multiplier = 1;
            X_pert_rem_sparse = multiplier * sign(X_pert_rem) .* (rand(B2,d) < (1/multiplier) * abs(X_pert_rem));
            X_pert(i:i2,:) = X_pert_round + X_pert_rem_sparse;
        end
    end
    if nnz(X_pert) > 0.5 * numel(X_pert)
        X_pert = full(X_pert);
    end
    % train SVM
    [lossPert, accPert, thetaPert] = trainMulticlass([X_train;X_pert], ([y_train;y_pert]+3)/2, 2, 0.005, 1e-4, N_train+N_pert, d, 10000);
    [lossPertBad, accPertBad] = testMulticlass(X_pert, (y_pert+3)/2, 2, N_pert, d, thetaPert, 2000);
    [lossPertTrain, accPertTrain] = testMulticlass(X_train, (y_train+3)/2, 2, N_train, d, thetaPert, 2000);
    thetaPert = thetaPert(:,2) - thetaPert(:,1);
    accGood = accPertTrain;
    accBad = accPertBad;
end