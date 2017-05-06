clear;
load process_data_results_enron_r50.mat;
opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'sedumi');
%%
[r_p,r_m] = deal(3,7);
[s_p,s_m] = deal(30,30);
%%
[mu_plus,mu_minus] = deal(mu_minus, mu_plus);
[r_p,r_m] = deal(r_m, r_p);
[s_p,s_m] = deal(s_m, s_p);

%%
epsilon = 0.25; eta = 0.02;
M_p = [];
M_m = [];
y_p = [];
y_m = [];
%M_p = [[1;0;-1;0] zeros(4,d-1)];
%M_m = [[0;1;0;-1] zeros(4,d-1)];
%y = [1;1;-1;-1];
%%
%theta_plus = 0.05 * ones(1,d);
%theta_minus = 0.05 * ones(1,d);
MAX_ITER=30; MAX_SLACK = 1.0;
slacks = zeros(MAX_ITER,2);
%for iter = 1:MAX_ITER
iter=1;
%%
M_p = [M_p; thetaPert'];
y_p = [y_p; 0];
M_m = [M_m; -thetaPert'];
y_m = [y_m; 0];

%%
while iter < 81
%
    if iter ~= 1 && slacks(iter-1,1) > MAX_SLACK
        M_p = [M_p; mean(sign(X_plus),2)'-1];
        y_p = [y_p; 0];
        M_p = [M_p; -(mean(X_plus,2)<-0.05)'];
        y_p = [y_p; 0];
    end
    if iter ~= 1 && slacks(iter-1,2) > MAX_SLACK
        M_m = [M_m; mean(sign(X_minus),2)'-1];
        y_m = [y_m; 0];
        M_m = [M_m; -(mean(X_minus,2)<-0.05)'];
        y_m = [y_m; 0];
    end
%

    
    [G, C, G_v, G_0, A, R_0, R_v, R, X_plus, X_minus] = sdpSmall(epsilon, mu_plus, mu_minus, p_plus, p_minus, r_p, r_m, s_p, s_m, opts, M_p, M_m, y_p, y_m, 50);
    %M_p = [M_p; -max(-X_plus', 0)/max(-X_plus'); zeros(1,d)];
    %M_m = [M_m; zeros(1,d); -max(-X_minus', 0)/max(-X_minus')];
    %y = [y; 0; 0];
    slacks(iter,1) = median(sum(abs(X_plus),1) - sum(X_plus,1));
    slacks(iter,2) = median(sum(abs(X_minus),1) - sum(X_minus,1));
    fprintf(1, 'slack: %.3f %.3f\n', slacks(iter,1), slacks(iter,2));
    X_plus_mean = mean(X_plus,2);
    X_minus_mean = mean(X_minus,2);
    [U,D,V] = svds(X_plus - X_plus_mean, 50);
    fprintf(1, 'norm: %.3f, variation: %.3f\n', norm(X_plus_mean, 2), sqrt(trace(D.^2)));
    [U,D,V] = svds(X_minus - X_minus_mean, 50);
    fprintf(1, 'norm: %.3f, variation: %.3f\n', norm(X_minus_mean, 2), sqrt(trace(D.^2)));
    %if slacks(iter,1) <= MAX_SLACK && slacks(iter,2) <= MAX_SLACK
    %    break;
    %end
    
%    theta_plus = theta_plus - eta * X_plus';
%    theta_minus = theta_minus - eta * X_minus';
%    fprintf(1, 'weights: [%.3f,%.3f,%.3f] [%.3f,%.3f,%.3f]\n', min(theta_plus), max(theta_plus), mean(abs(theta_plus)), min(theta_minus), max(theta_minus), mean(abs(theta_minus)));
%end
    iter=iter+1;
end
%%
xb_plus = mean(X_plus,2);
xb_minus = mean(X_minus,2);