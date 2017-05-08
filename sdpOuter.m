clear;
%load process_data_results_enron_r50.mat;
%load process_data_results_imdb_r50.mat;
load process_data_results_dogfish_r300.mat
opts = sdpsettings('verbose', 0, 'showprogress', 0, 'solver', 'sedumi');
%%
% enron
% [r_p,r_m] = deal(3,7);
% [s_p,s_m] = deal(30,30);
% imdb
% [r_p,r_m] = deal(18,16);
% [s_p,s_m] = deal(20,16);
% dogfish
 [r_p,r_m] = deal(25,25);
 [s_p,s_m] = deal(204,195);

%%
epsilon = 0.50;
MAX_SLACK = inf; %1.0;
M_p_theta = [];
M_m_theta = [];
y_p_theta = []; 
y_m_theta = [];
accGoods = [];
accBads = [];
nmlz = @(x) x/norm(x,2);
for major_iter = 1:8
    M_p_nn = [];
    y_p_nn = [];
    M_m_nn = [];
    y_m_nn = [];

    MAX_ITER = 100;
    slacks = zeros(MAX_ITER, 2);
    iter = 1;
    while iter < MAX_ITER
        if iter ~= 1 && slacks(iter-1,1) > MAX_SLACK
            M_p_nn = [M_p_nn; nmlz(mean(sign(X_plus),2)'-1)];
            y_p_nn = [y_p_nn; 0];
%            M_p_nn = [M_p_nn; -(mean(X_plus,2)<-0.05)'];
            M_p_nn = [M_p_nn; nmlz(mean(min(X_plus,0),2)')];
            y_p_nn = [y_p_nn; 0];
        end
        if iter ~= 1 && slacks(iter-1,2) > MAX_SLACK
            M_m_nn = [M_m_nn; nmlz(mean(sign(X_minus),2)'-1)];
            y_m_nn = [y_m_nn; 0];
%            M_m_nn = [M_m_nn; -(mean(X_minus,2)<-0.05)'];
            M_m_nn = [M_m_nn; nmlz(mean(min(X_minus,0),2)')];
            y_m_nn = [y_m_nn; 0];
        end
    
        fprintf(1, '\titeration %d\n', iter);
        [G, C, G_v, G_0, A, R_0, R_v, R, X_plus, X_minus] = sdpSmall(epsilon, mu_plus, mu_minus, p_plus, p_minus, r_p, r_m, s_p, s_m, opts, ... 
                                                                     [M_p_theta;M_p_nn], ...
                                                                     [M_m_theta;M_m_nn], ...
                                                                     [y_p_theta;y_p_nn], ...
                                                                     [y_m_theta;y_m_nn], 50);
        slacks(iter,1) = median(sum(abs(X_plus),1) - sum(X_plus,1));
        slacks(iter,2) = median(sum(abs(X_minus),1) - sum(X_minus,1));
        fprintf(1, '\t\tslack: %.3f %.3f\n', slacks(iter,1), slacks(iter,2));
        X_plus_mean = mean(X_plus,2);
        X_minus_mean = mean(X_minus,2);
        [U,D,V] = svds(X_plus - X_plus_mean, 50);
        fprintf(1, '\t\tnorm: %.3f, variation: %.3f\n', norm(X_plus_mean, 2), sqrt(trace(D.^2)));
        [U,D,V] = svds(X_minus - X_minus_mean, 50);
        fprintf(1, '\t\tnorm: %.3f, variation: %.3f\n', norm(X_minus_mean, 2), sqrt(trace(D.^2)));
        if max(slacks(iter,1), slacks(iter,2)) < MAX_SLACK
            break;
        end
        iter=iter+1;
    end
    
    xb_plus = mean(X_plus,2);
    xb_minus = mean(X_minus,2);
    
    [thetaPert, accGood, accBad, X_pert, y_pert] = trainPoisonedFnc(xb_plus, xb_minus, X_train, y_train, epsilon, p_plus, 0);
    accGoods = [accGoods accGood];
    accBads = [accBads accBad];
    disp(accGoods);
    disp(accBads);
%    if accGood < 0.55
%        break;
%    end
    fprintf(1, '\t<xb_p,theta> = %.4f\n', xb_plus' * thetaPert);
    fprintf(1, '\t<xb_m,theta> = %.4f\n', xb_minus' * thetaPert);
    if xb_plus' * thetaPert > 0
        M_p_theta = [M_p_theta; nmlz(thetaPert')];
        y_p_theta = [y_p_theta; 0];
    end
    if xb_minus' * thetaPert < 0
        M_m_theta = [M_m_theta; nmlz(-thetaPert')];
        y_m_theta = [y_m_theta; 0];
    end
    pause;
end
%%
xb_plus = mean(X_plus,2);
xb_minus = mean(X_minus,2);