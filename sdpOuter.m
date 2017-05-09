clear;
%load process_data_results_enron_r50.mat;
%load process_data_results_imdb_r50.mat;
%load process_data_results_dogfish_r300.mat
load process_data_results_mnist_r50.mat;
opts = sdpsettings('verbose', 2, 'showprogress', 1, 'solver', 'sedumi');
%X_train = X_proj';
%mu_plus = mu_proj_plus;
%mu_minus = mu_proj_minus;
%d = size(X_train, 2);
%%
% enron
% [r_p,r_m] = deal(3,7);
% [s_p,s_m] = deal(30,30);
% imdb
% [r_p,r_m] = deal(18,16);
% [s_p,s_m] = deal(20,16);
% dogfish
% r_sphere = [25 25];
% r_slab = [180 150];
% mnist
r_sphere = [4 6];
r_slab = [3.5 12];


probs = [p_plus p_minus];
%%
epsilon = 2.00;
MAX_SLACK = inf; %1.0;
M_theta = [];
y_theta = []; 
h_theta = [];
c_theta = [];
accGoods = [];
accBads = [];
losses = [];
lossesFull = [];
gradients = [];
thetas = [];
losses2 = [];
gradients2 = [];
w_noms = [];
xbps = [];
xbms = [];
nmlz = @(x) x/norm(x,2);
%

% M_theta = [M_theta; gnom'; zeros(1,d)'];
% y_theta = [y_theta; (L0 - Lnom) + gnom' * w_nom; gnom' * w_nom - Lnom];
% h_theta = [h_theta; 3; -1];
% c_theta = [c_theta; norm(gnom, 2); norm(thetaPert, 2)];

for major_iter = 1:100
    M_nn = [];
    y_nn = [];
    h_nn = [];
    c_nn = [];

    
    
    MAX_ITER = 100;
    slacks = zeros(MAX_ITER, 2);
    iter = 1;   
    while iter < MAX_ITER
        if iter ~= 1 && slacks(iter-1,1) > MAX_SLACK
            M_nn = [M_nn; nmlz(mean(sign(X_plus),2)'-1)];
            y_nn = [y_nn; 0];
            h_nn = [h_nn; 1];
            c_nn = [c_nn; 1];
            
            M_nn = [M_nn; nmlz(mean(min(X_plus,0),2)')];
            y_nn = [y_nn; 0];
            h_nn = [h_nn; 1];
            c_nn = [c_nn; 1];
        end
        if iter ~= 1 && slacks(iter-1,2) > MAX_SLACK
            M_nn = [M_nn; nmlz(mean(sign(X_minus),2)'-1)];
            y_nn = [y_nn; 0];
            h_nn = [h_nn; 2];
            c_nn = [c_nn; 1];

            M_nn = [M_nn; nmlz(mean(min(X_minus,0),2)')];
            y_nn = [y_nn; 0];
            h_nn = [h_nn; 2];
            c_nn = [c_nn; 1];
        end
    
        fprintf(1, '\titeration %d\n', iter);
        [G, C, G_v, A, X_plus, X_minus, W] = sdpSmall(epsilon, [mu_plus mu_minus], probs, r_sphere, r_slab, opts, ... 
                                                                     [M_theta;M_nn], ...
                                                                     [y_theta;y_nn], ...
                                                                     [h_theta;h_nn], ...
                                                                     [c_theta;c_nn], 2);
        slacks(iter,1) = median(sum(abs(X_plus),1) - sum(X_plus,1));
        slacks(iter,2) = median(sum(abs(X_minus),1) - sum(X_minus,1));
        %fprintf(1, '\t\tslack: %.3f %.3f\n', slacks(iter,1), slacks(iter,2));
        X_plus_mean = mean(X_plus,2);
        X_minus_mean = mean(X_minus,2);
%        [U,D,V] = svds(X_plus - X_plus_mean, 50);
        fprintf(1, '\t\tnorm: %.3f, variation: %.3f\n', norm(X_plus_mean, 2), norm(X_plus-X_plus_mean, 'fro')); %sqrt(trace(D.^2)));
%        [U,D,V] = svds(X_minus - X_minus_mean, 50);
        fprintf(1, '\t\tnorm: %.3f, variation: %.3f\n', norm(X_minus_mean, 2), norm(X_minus-X_minus_mean, 'fro')); %sqrt(trace(D.^2)));
        if max(slacks(iter,:)) < MAX_SLACK
            break;
        end
        iter=iter+1;
    end
    
    xb_plus = X_plus(:,1); %mean(X_plus,2);
    xb_minus = X_minus(:,1); %mean(X_minus,2);
    w_nom = W(:,1);
    
    thetaPert = 0;
    accGood = 0;
    accBad = 0;
    numTrials = 1;
    for jj=1:numTrials
        [thetaPert_cur, accGood_cur, accBad_cur, X_pert, y_pert] = trainPoisonedFnc(xb_plus, xb_minus, X_train, y_train, epsilon, p_plus, 0);
        thetaPert = thetaPert + thetaPert_cur / numTrials;
        accGood = accGood + accGood_cur / numTrials;
        accBad = accBad + accBad_cur / numTrials;
    end
    accGoods = [accGoods accGood];
    accBads = [accBads accBad];
    xbps = [xbps; xb_plus'];
    xbms = [xbms; xb_minus'];
    accGoods
    accBads
%    if accGood < 0.55
%        break;
%    end
    fprintf(1, '\t<xb_p,theta> = %.4f\n', xb_plus' * thetaPert);
    fprintf(1, '\t<xb_m,theta> = %.4f\n', xb_minus' * thetaPert);
    [g,L0] = nabla_Loss(X_train, y_train, thetaPert, 0);
    losses = [losses L0];
    losses
    L0full = L0 + epsilon * p_plus * max(1 - xb_plus' * thetaPert, 0) ...
                + epsilon * p_minus * max(1 + xb_minus' * thetaPert, 0);
    lossesFull = [lossesFull L0full];
    gradients = [gradients; g'];
    thetas = [thetas; thetaPert'];
    lossesFull
    [gnom,Lnom] = nabla_Loss(X_train, y_train, w_nom, 0);
    losses2 = [losses2 Lnom];
    gradients2 = [gradients2; gnom'];
    w_noms = [w_noms; w_nom'];
    Lnom_extra = epsilon * (1 - p_plus * xb_plus' * w_nom + p_minus * xb_minus' * w_nom);
    if major_iter > 1
%         Lnom_pred = losses(end-1) + gradients(end-1,:) * (w_nom - thetas(end-1,:)');
%         fprintf(1, 'predicted loss of nominal w: %.4f / %.4f\n', Lnom_pred, Lnom_pred + Lnom_extra);
%         Ltheta_pred = losses(end-1);
%         Ltheta_extra = epsilon * (1 - p_plus * xb_plus' * thetas(end-1,:)' + p_minus * xb_minus' * thetas(end-1,:)');
%         fprintf(1, 'predicted loss of previous theta: %.4f / %.4f\n', Ltheta_pred, Ltheta_pred + Ltheta_extra);
        Lnom_pred = losses2(end-1) + gradients2(end-1,:) * (w_nom - w_noms(end-1,:)');
        fprintf(1, 'predicted loss of nominal w: %.4f / %.4f\n', Lnom_pred, Lnom_pred + Lnom_extra);
    end
%     if true
%         Lnom_pred_cur = losses(end) + gradients(end,:) * (w_nom - thetas(end,:)');
%         fprintf(1, 'predicted loss of nominal w: %.4f / %.4f (current constraint)\n', Lnom_pred_cur, Lnom_pred_cur + Lnom_extra);
%     end
    fprintf(1, '   actual loss of nominal w: %.4f / %.4f\n', Lnom, Lnom + Lnom_extra);
    %if major_iter > 1 && abs(Lnom_pred - Lnom) < 0.2
    %    pause;
    %end
    if true || xb_plus' * thetaPert < 0.4 || xb_minus' * thetaPert > -0.4
        disp('adding constraint for w...');
        % need to make sure that L(w) + epsilon * (p_plus * max(1-<xb_plus,w>) + p_minus * max(1+<xb_minus,w>)) <= corresponding value at w0
%         if xb_plus' * thetaPert <= 1
%             g = g - epsilon * p_plus * xb_plus;
%         end
%         if xb_minus' * thetaPert >= -1
%             g = g + epsilon * p_minus * xb_minus;
%         end
%        norm_g = norm(g,2);
        % TODO this is actually kind of loose, since uses the stale xb_plus
        %M_theta = [M_theta; g'];
        %y_theta = [y_theta; (L0full-L0) - epsilon + g' * thetaPert];
        %h_theta = [h_theta; 3];
        
        M_theta = [M_theta; gnom'; thetaPert'];
        y_theta = [y_theta; (L0 - Lnom) + gnom' * w_nom; gnom' * w_nom - Lnom];
        h_theta = [h_theta; 3; -1];
        c_theta = [c_theta; norm(gnom, 2); norm(thetaPert, 2)];
    end
    if accGood(1) > 0.6 && xb_plus' * thetaPert > 0
        disp('adding constraint for x_plus...');
        M_theta = [M_theta; nmlz(thetaPert')];
        y_theta = [y_theta; 0];
        h_theta = [h_theta; 1];
        c_theta = [c_theta; 1];
    end
    if accGood(2) > 0.6 && xb_minus' * thetaPert < 0
        disp('adding constraint for x_minus...');
        M_theta = [M_theta; nmlz(-thetaPert')];
        y_theta = [y_theta; 0];
        h_theta = [h_theta; 2];
        c_theta = [c_theta; 1];
    end
end
%%
xb_plus = mean(X_plus,2);
xb_minus = mean(X_minus,2);