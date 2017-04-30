%load Q_agg.mat;
%% set up quadratic program
disp('setting up variables...');
tic;
epsilon = 0.04; %0.03;
t = 1.3;
rho = 2.4;
sig_half_w = sdpvar(r, 1);
a = (1/sqrt(2*pi)) * exp(-0.5 * (1 + t)^2 / rho^2) / rho;
b = normcdf((1+t)/rho);
M = ((1-epsilon) / epsilon) * a * Q_agg * V * D;
z = (((1-epsilon) / epsilon) * b + 1) * Q_agg * mu;
residual = M * sig_half_w - z;
Objective = residual' * residual;
mu_transpose_w = (mu'*V)/D * sig_half_w;
toc;
%%
disp('setting up margin variables...');
tic;
dd = 0;
indices = 1:dd;
margin = sdpvar(dd,1);
if dd ~= 0
  xb_s = ((1-epsilon) / epsilon) * (a * V * D * sig_half_w - b * mu);
end
toc;
disp('Setting up optimization...');
tic;
Constraint = [mu_transpose_w == 1; sig_half_w' * sig_half_w <= rho^2; -mu_all(indices)-margin <= xb_s(indices) <= mu_all(indices)+margin; margin >= 0; ones(1,dd) * margin <= 30]; % + xb_s >= 0; mu_all - xb_s >= 0];
toc;
%% solve QP
optimize(Constraint, Objective);
%% extract solution
disp('Extracting solution...');
tic;
w = V/D * double(sig_half_w);
xb = ((1-epsilon) / epsilon) * (a * V * D * double(sig_half_w) - b * mu);
toc;
disp('Results:');
fprintf(1, '\tObjective value: %.4f\n', double(Objective));
fprintf(1, '\trho: %.4f (vs %.4f)\n', sqrt(double(sig_half_w' * sig_half_w)), rho);
