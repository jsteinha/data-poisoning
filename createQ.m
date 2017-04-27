%% construct Q matrix
Qs = cell(4,1);
Qs{1} = speye(d); % d x d
Qs{2} = mu'; % 1 x d
r = 200;
[U,D,V] = svds(A / sqrt(N), r);
% want pseudo-inverse
% A = U * D * V'
% Sigma = V * D^2 * V' / N
% Sigma_pinv = V * D^(-2) * V'
Qs{3} = D\V'; % r x d
Qs{4} = ones(1,d); % 1 x d
%
taus = cell(4,1);
for i=1:4
    taus{i} = zeros(N,1);
    for j=1:N
        if mod(j,100) == 0
            fprintf(1, 'i=%d, j=%d\n', i, j);
        end
        taus{i}(j) = norm(Qs{i}*(A(j,:)' - mu_all - yt(j) * mu), 2);
    end
end
%
thresholds = zeros(4,1);
for i=1:4
    thresholds(i) = quantile(taus{i}, 0.75);
end
%
Q_agg = sparse([]); % r_agg x d
for i=1:4
    Q_agg = [Q_agg; sparse(Qs{i} / thresholds(i))];
end
%
taus_agg = zeros(N,1);
for j=1:N
    if mod(j,100) == 0
        fprintf(1, 'i=agg, j=%d\n', j);
    end
    taus_agg(j) = norm(Q_agg*(A(j,:)' - mu_all - yt(j) * mu), 2);
end
save Q_agg.mat;
%% set up quadratic program
epsilon = 0.04; %0.03;
t = 1;
rho = sqrt(5.5);
sig_half_w = sdpvar(r, 1);
a = (1/sqrt(2*pi)) * exp(-0.5 * (1 + t)^2 / rho^2) / rho;
b = normcdf((1+t)/rho);
M = ((1-epsilon) / epsilon) * a * Q_agg * V * D;
z = (((1-epsilon) / epsilon) * b + 1) * Q_agg * mu;
residual = M * sig_half_w - z;
Objective = residual' * residual;
mu_transpose_w = (mu'*V)/D * sig_half_w;
xb_s = ((1-epsilon) / epsilon) * (a * V * D * sig_half_w - b * mu);
%%
dd = 150;
indices = 1:dd;
margin = sdpvar(dd,1);
Constraint = [mu_transpose_w == 1; sig_half_w' * sig_half_w <= rho^2; -mu_all(indices)-margin <= xb_s(indices) <= mu_all(indices)+margin; margin >= 0; ones(1,dd) * margin <= 8]; % + xb_s >= 0; mu_all - xb_s >= 0];
%% solve QP
optimize(Constraint, Objective);
%% extract solution
w = V/D * double(sig_half_w);
xb = ((1-epsilon) / epsilon) * (a * V * D * double(sig_half_w) - b * mu);