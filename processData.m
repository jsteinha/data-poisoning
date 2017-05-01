function processData(X_train, y_train, X_test, y_test, name, r)
  N_train = size(X_train, 1);
  N_test = size(X_test, 1);
  d = size(X_train, 2);
  assert(d == size(X_test, 2));
  assert(N_train == size(y_train,1));
  assert(N_test == size(y_test,1));
  assert(all(y_train == 1 | y_train == -1));
  assert(all(y_test == 1 | y_test == -1));
  if nargin < 6
    r = 50;
    fprintf(1, 'no rank specified, defaulting to %d\n', r);
  end
  disp('Computing initial SVD...');
  tic;
  [~,~,V] = svds(X_train, r);
  toc;
  N_plus = sum(y_train == 1);
  N_minus = sum(y_train == -1);
  mu_plus = X_train' * (y_train == 1) / N_plus;
  mu_minus = X_train' * (y_train == -1) / N_minus;
  disp('Computing QR decomposition');
  tic;
  size(V)
  size(mu_plus)
  size(mu_minus)
  [Q,~] = qr([V mu_plus mu_minus], 0);
  toc;
  Q = Q'; % Q is now d_proj x d
  d_proj = size(Q,1);
  size(Q)
  clear V;
  % we will work in the subspace defined by V
  X_proj = Q * X_train';
  mu_proj = X_proj * ones(N_train,1) / N_train;
  mu_proj_plus = X_proj * (y_train == 1) / N_plus;
  mu_proj_minus = X_proj * (y_train == -1) / N_minus;
  p_plus = N_plus / N_train;
  p_minus = N_minus / N_train;
  Sigma_proj_plus = cov(X_proj(:,logical(y_train==1))', 0);
  Sigma_proj_minus = cov(X_proj(:,logical(y_train==-1))', 0);
  assert(norm(Q*mu_plus - mu_proj_plus, 2) < 1e-8);
  assert(norm(Q*mu_minus - mu_proj_minus, 2) < 1e-8);

  % potential function is x' * Psi * x
  % Psi_half is such that Psi = Psi_half' * Psi_half
  C = 4;
  Psi_half_init = cell(C,2);
  Psi_half_init{1,1} = eye(d_proj);
  Psi_half_init{1,2} = eye(d_proj); 
  Psi_half_init{2,1} = inv(Sigma_proj_plus);
  Psi_half_init{2,2} = inv(Sigma_proj_minus);
  Psi_half_init{3,1} = (mu_plus - mu_minus)' * Q';
  Psi_half_init{3,2} = (mu_plus - mu_minus)' * Q';
  Psi_half_init{4,1} = ones(1,d) * Q';
  Psi_half_init{4,2} = ones(1,d) * Q';

  taus = cell(C,2);
  N0  = min([1000, N_plus, N_minus]);
  I_plus  = 1:N_train; I_plus  = I_plus(logical(y_train == 1));
  Pi = randperm(length(I_plus)); I_plus  = I_plus(Pi(1:N0));
  I_minus = 1:N_train; I_minus = I_minus(logical(y_train == -1)); 
  Pi = randperm(length(I_minus)); I_minus = I_minus(Pi(1:N0));
  clear Pi;
  disp('Computing taus...');
  tic;
  for i=1:C
    taus{i,1} = zeros(N0,1);
    taus{i,2} = zeros(N0,1);
    for j=1:N0
      if mod(j,100) == 0
        fprintf(1, 'i=%d, j=%d\n', i, j);
      end
      taus{i,1}(j) = norm(Psi_half_init{i,1} * (X_proj(:,I_plus(j)) - mu_proj_plus), 2);
      taus{i,2}(j) = norm(Psi_half_init{i,2} * (X_proj(:,I_minus(j)) - mu_proj_minus), 2);
    end
  end
  disp('Done computing taus');
  toc;

  thresholds = zeros(C,2);
  disp('Computing thresholds');
  for i=1:C
    t1 = quantile(taus{i,1}, 0.75);
    t2 = quantile(taus{i,2}, 0.75);
    fprintf(1, '\tplus(%d): %8.4f', i, t1);
    fprintf(1, '\tminus(%d): %8.4f\n', i, t2);
    %thresholds(i) = min(t1, t2); %quantile(taus{i,1}, 0.75), quantile(taus{i,2}, 0.75));
    thresholds(i,1) = t1;
    thresholds(i,2) = t2;
  end

  Psi_half = cell(2,1);
  Psi_half{1} = [];
  Psi_half{2} = [];
  for i=1:C
    for j=1:2
      Psi_half{j} = [Psi_half{j}; Psi_half_init{i,j} / thresholds(i,j)];
    end
  end

  disp('Computing tau_agg');
  tic;
  tau_agg = zeros(N0,2);
  for j=1:N0
    if mod(j,100) == 0
      fprintf(1, 'i=agg, j=%d\n', j);
    end
    tau_agg(j,1) = norm(Psi_half{1} * (X_proj(:, I_plus(j)) - mu_proj_plus), 2);
    tau_agg(j,2) = norm(Psi_half{2} * (X_proj(:, I_minus(j)) - mu_proj_minus), 2);
  end
  toc;
  fprintf('tau_agg 0.75 quantiles:\n\tplus: %.4f\n\tminus: %.4f\n', quantile(tau_agg(:,1), 0.75), quantile(tau_agg(:,2), 0.75));
  save(sprintf('process_data_results_%s_r%d.mat', name, r));
end
