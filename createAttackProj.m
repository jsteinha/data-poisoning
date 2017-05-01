function [rho_plus, rho_minus, xb_plus, xb_minus] = createAttackProj(t, rho_plus, rho_minus, epsilon, Psi_half, Sigma_proj_plus, Sigma_proj_minus, mu_proj_plus, mu_proj_minus, p_plus, p_minus, Q, enforce_pos)
  d_proj = size(Q,1);
  d = size(Q,2);
  disp('setting up variables...');
  tic;
  a_plus  = (1/sqrt(2*pi)) * exp(-0.5 * (1 + t)^2 / rho_plus^2)  / rho_plus;
  a_minus = (1/sqrt(2*pi)) * exp(-0.5 * (1 + t)^2 / rho_minus^2) / rho_minus;
  b_plus  = normcdf((1+t)/rho_plus);
  b_minus = normcdf((1+t)/rho_minus);

  w0 = sdpvar(d_proj, 1);
  if enforce_pos
    xb_plus_pre = sdpvar(d, 1);
    xb_minus_pre = sdpvar(d, 1);
    xb_plus = Q * xb_plus_pre;
    xb_minus = Q * xb_minus_pre;
  else
    xb_plus  = sdpvar(d_proj, 1);
    xb_minus = sdpvar(d_proj, 1);
  end
  M_plus  = a_plus  * Sigma_proj_plus;
  M_minus = a_minus * Sigma_proj_minus;
  z_plus  = b_plus  * mu_proj_plus;
  z_minus = b_minus * (-mu_proj_minus);
  partial_L = (p_plus*M_plus+p_minus*M_minus) * w0 - (p_plus*z_plus + p_minus*z_minus);
  residual_plus  = Psi_half{1} * (xb_plus - mu_proj_plus);
  residual_minus = Psi_half{2} * (xb_minus - mu_proj_minus);
  max_err = sdpvar(1,1);
  mu_transpose_w = w0' * (p_plus * mu_proj_plus - p_minus * mu_proj_minus);
  rho_plus_squared_sym  = w0' * Sigma_proj_plus * w0;
  rho_minus_squared_sym = w0' * Sigma_proj_minus * w0;
  toc;

  disp('setting up optimization...');
  tic;
  Objective = max_err;
  Constraint = [mu_transpose_w == -t; 
                p_plus * xb_plus - p_minus * xb_minus == (1-epsilon) / epsilon * partial_L;
                residual_plus' * residual_plus <= max_err; 
                residual_minus' * residual_minus <= max_err; 
                rho_plus_squared_sym <= rho_plus^2; 
                rho_minus_squared_sym <= rho_minus^2];
  if enforce_pos
    Constraint = [Constraint; 
                  xb_plus_pre >= 0;
                  xb_minus_pre >= 0];
  end
  toc;

  disp('solving optimization...');
  optimize(Constraint, Objective);

  disp('Extracting solution...');
  tic;
  w = Q' * double(w0);
  rho_plus = sqrt(double(rho_plus_squared_sym));
  rho_minus = sqrt(double(rho_minus_squared_sym));
  if enforce_pos
    xb_plus = double(xb_plus_pre);
    xb_minus = double(xb_minus_pre);
  else
    xb_plus = Q' * double(xb_plus);
    xb_minus = Q' * double(xb_minus);
  end
  disp('Results:');
  fprintf(1, '\tObjective value: %.4f\n', sqrt(double(Objective)));
  fprintf(1, '\trho_+: %.4f\n', rho_plus);
  fprintf(1, '\trho_-: %.4f\n', rho_minus);
end
