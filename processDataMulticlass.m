function processData(X_train, y_train, X_test, y_test, name, r, k)
  N_train = size(X_train, 1);
  N_test = size(X_test, 1);
  d = size(X_train, 2);
  assert(d == size(X_test, 2));
  assert(N_train == size(y_train,1));
  assert(N_test == size(y_test,1));
  if k == 2
    assert(all(y_train == 1 | y_train == -1));
    assert(all(y_test == 1 | y_test == -1));
  else
    assert(all(1 <= y_train & y_train <= k));
    assert(all(1 <= y_test & y_test <= k));
  end
  if nargin < 6
    r = 50;
    fprintf(1, 'no rank specified, defaulting to %d\n', r);
  end
  disp('Computing initial SVD...');
  tic;
  [~,~,V] = svds(X_train, r);
  toc;
  Ns = zeros(k,1);
  mus = zeros(d,k);
  for j=1:k
    Ns(j) = sum(y_train == j);
    mus(:,j) = X_train' * (y_train == j) / Ns(j);
  end
  disp('Computing QR decomposition');
  tic;
  [Q,~] = qr([V mus], 0);
  toc;
  Q = Q'; % Q is now d_proj x d
  d_proj = size(Q,1);
  size(Q)
  clear V;
  % we will work in the subspace defined by V
  X_proj = Q * X_train';
  mu_proj_all = X_proj * ones(N_train,1) / N_train;
  mu_projs = zeros(d_proj,k);
  for j=1:k
    mu_projs(:,j) = X_proj * (y_train == j) / Ns(j);
  end
  ps = Ns / N_train;
  Sigma_projs = cell(k,1);
  for j=1:k
    Sigma_projs{j} = cov(X_proj(:,logical(y_train==j))', 0);
  end
  for j=1:k
    assert(norm(Q*mus(:,j) - mu_projs(:,j), 2) < 1e-8);
  end

  % potential function is x' * Psi * x
  % Psi_half is such that Psi = Psi_half' * Psi_half
  C = 3+k;
  Psi_half_init = cell(C,k);
  for j=1:k
    Psi_half_init{1,j} = eye(d_proj);
    Psi_half_init{2,j} = inv(Sigma_projs{j});
    Psi_half_init{3,j} = ones(1,d) * Q';
    for j2=1:k
      if j==j2
        Psi_half_init{3+j2,j} = mus(:,j)' * Q';
      else
        Psi_half_init{3+j2,j} = (mus(:,j) - mus(:,j2))' * Q';
      end
    end
  end

  taus = cell(C,k);
  N0  = min([1000, min(Ns)]);
  Is = cell(k,1);
  for j=1:k
    Is{j} = 1:N_train; Is{j} = Is{j}(logical(y_train == j));
    Pi = randperm(length(Is{j})); Is{j} = Is{j}(Pi(1:N0));
  end
  clear Pi;
  disp('Computing taus...');
  tic;
  for i=1:C
    for j=1:k
      taus{i,j} = zeros(N0,1);
    end
    for n=1:N0
      if mod(n,100) == 0
        fprintf(1, 'i=%d, n=%d\n', i, n);
      end
      for j=1:k
        taus{i,j}(n) = norm(Psi_half_init{i,j} * (X_proj(:,Is{j}(n)) - mu_projs(:,j)), 2);
      end
    end
  end
  disp('Done computing taus');
  toc;

  thresholds = zeros(C,k);
  disp('Computing thresholds');
  for i=1:C
    for j=1:k
      tj = quantile(taus{i,j}, 0.75);
      fprintf(1, '\t(%d,%d): %8.4f', i, j, tj);
      thresholds(i,j) = tj;
    end
    fprintf(1,'\n');
  end

  Psi_half = cell(k,1);
  for j=1:k
    Psi_half{j} = [];
    for i=1:C
      Psi_half{j} = [Psi_half{j}; Psi_half_init{i,j} / thresholds(i,j)];
    end
  end

  disp('Computing tau_agg');
  tic;
  tau_agg = zeros(N0,k);
  for n=1:N0
    if mod(n,100) == 0
      fprintf(1, 'i=agg, n=%d\n', n);
    end
    for j=1:k
      tau_agg(n,j) = norm(Psi_half{j} * (X_proj(:, Is{j}(n)) - mu_projs(:,j)), 2);
    end
  end
  toc;
  fprintf('tau_agg 0.75 quantiles:\n');
  for j=1:k
    fprintf('\t%d: %.4f\n', j, quantile(tau_agg(:,j), 0.75));
  end
  save(sprintf('process_data_results_%s_r%d_k%d.mat', name, r, k));
end
