function [thetaPert, accGood, accBad, X_pert, y_pert] = trainPoisonedFnc(xb_plus, xb_minus, X_train, y_train, epsilon, p_plus, dataOpts)
    N_train = size(X_train, 1);
    d = size(X_train, 2);
    N_pert = round(epsilon * N_train);
    y_pert = 2*(rand(N_pert,1)<p_plus)-1;
    if dataOpts.sparse
        X_pert = sparse(N_pert,d);
    else
        X_pert = zeros(N_pert, d);
    end
    B = 100;
    for i=1:B:N_pert
        %i
        i2 = min(i+B-1,N_pert);
        B2 = i2-i+1;
        X_pert_cur = (y_pert(i:i2)==1) * xb_plus' + (y_pert(i:i2)==-1) * xb_minus';
        if isfield(dataOpts, 'lower')
            X_pert_cur = max(dataOpts.lower, X_pert_cur);
        end
        if isfield(dataOpts, 'upper')
            X_pert_cur = min(dataOpts.upper, X_pert_cur);
        end
        if dataOpts.integer
            X_pert_round = round(X_pert_cur);
            X_pert_rem = X_pert_cur - X_pert_round;
            multiplier = 1;
            X_pert_rem_sparse = multiplier * sign(X_pert_rem) .* (rand(B2,d) < (1/multiplier) * abs(X_pert_rem));
            X_pert(i:i2,:) = X_pert_round + X_pert_rem_sparse;
        else
            X_pert(i:i2,:) = X_pert_cur;
        end
    end
    % train SVM
    [lossPert, accPert, thetaPert] = trainMulticlass([X_train;X_pert], ([y_train;y_pert]+3)/2, 2, 0.005, 1e-4, N_train+N_pert, d, 3000, 1);
    [lossPertBad, accPertBad] = testMulticlass(X_pert, (y_pert+3)/2, 2, N_pert, d, thetaPert, 2000, 1);
    [lossPertTrain, accPertTrain] = testMulticlass(X_train, (y_train+3)/2, 2, N_train, d, thetaPert, 2000, 1);
    thetaPert = thetaPert(:,2) - thetaPert(:,1);
    accGood = accPertTrain;
    accBad = accPertBad;
end