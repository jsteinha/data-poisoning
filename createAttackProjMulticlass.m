function [ts, rhos, xb_full, p_attack, w] = createAttackProjMulticlass(k, epsilon, j_attack, j_target, ts, rhos, Psi_half, Sigma_projs, mu_projs, ps, Q)
  d_proj = size(Q,1);
  d = size(Q,2);
  disp('setting up variables...');

  disp('declaring w0, xb...');
  tic;

  max_err = sdpvar(1,1);
  Objective = max_err; 
  Constraint = [];

  w0 = cell(k,1);
  xb = cell(k,1);
  for j=1:k
    xb{j} = sdpvar(d_proj, 1);
    w0{j} = sdpvar(d_proj, 1);
  end

  toc;

  disp('constraining residuals...');
  tic;

  residual = cell(k,1);
  for j=1:k
    residual{j} = Psi_half{j} * (xb{j} - mu_projs(:,j));
    Constraint = [Constraint; residual{j}' * residual{j} <= max_err];
  end

  toc;

  disp('building partial_L and rho_squared_sym...');
  tic;

  partial_L = cell(k,1);
  rho_squared_sym = cell(k,k);
  lambda = sdpvar(d_proj,1); % Lagrange multiplier since \sum_j w_j = 0
  parfor j=1:k
    partial_L{j} = lambda;
    for j2=1:k
      if j ~= j2
%dbstop if error
        partial_L{j} = partial_L{j} + ps(j) * (exp(-0.5*(1+ts(j,j2))^2/rhos(j,j2)^2) * Sigma_projs{j} * (w0{j} - w0{j2}) / rhos(j,j2) - normcdf((1+ts(j,j2))/rhos(j,j2)) * mu_projs(:,j));
        rho_squared_sym{j,j2} = (w0{j}-w0{j2})' * Sigma_projs{j} * (w0{j}-w0{j2});
        %Constraint = [Constraint; rho_squared_sym{j,j2} <= rhos(j,j2)^2];
      end
    end
  end
  for j=1:k
    for j2=1:k
      if j ~= j2
        Constraint = [Constraint; rho_squared_sym{j,j2} <= rhos(j,j2)^2];
      end
    end
  end

  toc;

  disp('constraining mu_transpose_w');
  tic;

  for j=1:k
    for j2=1:k
      if j ~= j2
        %mu_transpose_w = mu_projs(:,j_attack)' * (w0{j_attack} - w0{j_target});
        mu_transpose_w = mu_projs(:,j)' * (w0{j} - w0{j2});
        if j == j_attack && j2 == j_target
          Constraint = [Constraint; mu_transpose_w <= -ts(j,j2)];
        else
          Constraint = [Constraint; mu_transpose_w >= -ts(j,j2)];
        end
      end
    end
  end

  toc;

  disp('constraining sum_j w_j = 0');
  tic;

  w_sum = 0;
  for j=1:k
    w_sum = w_sum + w0{j};
  end
  Constraint = [Constraint; w_sum == 0];

  toc;

  disp('constraining xb = partial_L');
  tic;

  p_attack = (ps + (1:k==j_attack)' + (1:k==j_target)')/3.0;
  for j=1:k
    Constraint = [Constraint; p_attack(j) * xb{j} == (1 / epsilon) * partial_L{j}];
  end
  %Constraint = [Constraint; xb_mean == (1-epsilon) / epsilon * partial_L];

  toc;

  disp('solving optimization...');
  optimize(Constraint, Objective);

  disp('Extracting solution...');
  tic;
  w = cell(k,1);
  for j=1:k
    w{j} = Q' * double(w0{j});
  end

  for j=1:k
    for j2=1:k
      if j ~= j2
        rhos(j,j2) = sqrt(double(rho_squared_sym{j,j2}));
        ts(j,j2) = -mu_projs(:,j)' * double(w0{j} - w0{j2});
      end
    end
  end

  xb_full = zeros(d, k);
  for j=1:k
    xb_full(:,j) = Q' * double(xb{j}); 
  end

  disp('Results:');
  fprintf(1, '\tObjective value: %.4f\n', sqrt(double(Objective)));
  for j=1:k
    for j2=1:k
      if j ~= j2
        fprintf(1, '\trho(%d,%d): %.4f, t(%d,%d) = %.4f\n', j, j2, rhos(j,j2), j, j2, ts(j,j2));
      end
    end
  end
end
