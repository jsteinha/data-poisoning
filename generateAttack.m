function [xb_plus, xb_minus, w_nom] = generateAttack(epsilon, mu, probs, r_sphere, r_slab, M_theta, y_theta, h_theta, c_theta, opts, dataOpts)
    MAX_ITER = 100;
    slackLower = zeros(MAX_ITER, 2);
    slackUpper = zeros(MAX_ITER, 2);
    iter = 1;
    
    M_nn = [];
    y_nn = [];
    h_nn = [];
    c_nn = [];
    nmlz = @(x) x/norm(x,2);

    while iter <= MAX_ITER
        if iter ~= 1
            if isfield(dataOpts, 'lower')
                for j=1:2
                    if slacksLower(iter-1,j) > MAX_SLACK
                        v = mean(sign(X_cell{j}-dataOpts.lower),2)'-1;
                        y = sum(v) * dataOpts.lower;
                        M_nn = [M_nn; v];
                        y_nn = [y_nn; y];
                        h_nn = [h_nn; j];
                        c_nn = [c_nn; norm(v,2)];

                        v = mean(min(X_cell{j}-dataOpts.lower,0),2)';
                        y = sum(v) * dataOpts.lower;
                        M_nn = [M_nn; v];
                        y_nn = [y_nn; y];
                        h_nn = [h_nn; j];
                        c_nn = [c_nn; norm(v,2)];
                    end
                end
            end
            if isfield(dataOpts, 'upper')
                for j=1:2
                    if slacksUpper(iter-1,j) > MAX_SLACK
                        v = mean(sign(X_cell{j}-dataOpts.upper),2)'+1;
                        y = sum(v) * dataOpts.upper;
                        M_nn = [M_nn; v];
                        y_nn = [y_nn; y];
                        h_nn = [h_nn; j];
                        c_nn = [c_nn; norm(v,2)];

                        v = mean(max(X_cell{j}-dataOpts.upper,0),2)';
                        M_nn = [M_nn; v];
                        y_nn = [y_nn; y];
                        h_nn = [h_nn; j];
                        c_nn = [c_nn; norm(v,2)];
                    end
                end
            end
        end
    
        fprintf(1, '\titeration %d\n', iter);
        [G, C, G_v, A, X_plus, X_minus, W] = sdpSmall(epsilon, mu, probs, r_sphere, r_slab, opts, ... 
                                                                     [M_theta;M_nn], ...
                                                                     [y_theta;y_nn], ...
                                                                     [h_theta;h_nn], ...
                                                                     [c_theta;c_nn], 50);
        allOk = true;
        X_cell = {X_plus X_minus};
        if isField(dataOpts, 'lower')
            for j=1:2
                slackLower(iter,j) = median(sum(max(dataOpts.lower-X_cell{j}, 0),1));
            end
            fprintf(1, '\t\tslackLower: %.3f %.3f\n', slackLower(iter,1), slackLower(iter,2));
            if max(slackLower(iter,:)) >= MAX_SLACK
                allOk = false;
            end
        end
        if isField(dataOpts, 'upper')
            for j=1:2
                slackUpper(iter,j) = median(sum(max(X_cell{j}-dataOpts.upper, 0),1));
            end
            fprintf(1, '\t\tslackLower: %.3f %.3f\n', slackUpper(iter,1), slackUpper(iter,2));
            if max(slackUpper(iter,:)) >= MAX_SLACK
                allOk = false;
            end
        end
        X_plus_mean = mean(X_plus,2);
        X_minus_mean = mean(X_minus,2);
        fprintf(1, '\t\tnorm: %.3f, variation: %.3f\n', norm(X_plus_mean, 2), norm(X_plus-X_plus_mean, 'fro'));
        fprintf(1, '\t\tnorm: %.3f, variation: %.3f\n', norm(X_minus_mean, 2), norm(X_minus-X_minus_mean, 'fro'));
        if allOk
            break;
        end
        iter=iter+1;
    end
    
    xb_plus = X_plus(:,1);
    xb_minus = X_minus(:,1);
    w_nom = W(:,1);
end