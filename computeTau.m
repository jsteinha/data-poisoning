function taus = computeTau(X, y, tau_fnc)
  N = length(y);
  if size(X,2) ~= N
    X = X';
    assert(size(X,2) == N);
  end
  N_plus = sum(y==1); N_minus = sum(y==-1);
  N0  = min([1000, N_plus, N_minus]);
  Pi = randperm(N0);
  I_plus  = 1:N; I_plus  = I_plus(logical(y == 1));   I_plus  = I_plus(Pi(1:N0));
  Pi = randperm(N0);
  I_minus = 1:N; I_minus = I_minus(logical(y == -1)); I_minus = I_minus(Pi(1:N0));
  taus = zeros(N0,2);
  disp('Computing taus...');
  tic;
  for j=1:N0
    if mod(j,100) == 0
      fprintf(1, 'j=%d\n', j);
    end
    taus(j,1)= tau_fnc(X(:,I_plus(j)), y(I_plus(j)));
    taus(j,2)= tau_fnc(X(:,I_minus(j)), y(I_minus(j)));
  end
  disp('Done computing taus');
  toc;
end
