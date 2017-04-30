tau_fnc = @(x,y) norm(Psi_half * (Q * (x-(y==1) * mu_plus - (y==-1) * mu_minus)), 2);
%tau_fnc = @(x,y) sum(x~=0);
%tau_fnc = @(x,y) norm(x,1);
taus_train = computeTau(X_train, y_train, tau_fnc);
taus_pert = computeTau(X_pert, y_pert, tau_fnc);
for c=1:2
  x_min = min([0, min(taus_train(:,c)), min(taus_pert(:,c))]);
  x_max = max(max(taus_train(:,c)), max(taus_pert(:,c)));
  x_step = (x_max-x_min) / 20;
  figure(c); clf;
  histogram(taus_train(:,c), x_min:x_step:x_max);
  h = findobj(gca,'Type','patch');
  hold on;
  histogram(taus_pert(:,c), x_min:x_step:x_max);
  set(findobj(gca,'Type','patch'),'FaceColor','w','EdgeColor','r');
  set(h,'EdgeColor','b');
end
