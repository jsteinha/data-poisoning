load imdb.mat;
path(pathdef);
%% construct Q matrix
Qs = cell(4,1);
Qs{1} = speye(d); % d x d
Qs{2} = mu'; % 1 x d
r = 500;
disp('Computing SVD...');
tic;
[U,D,V] = svds(A / sqrt(N), r);
toc;
% want pseudo-inverse
% A = U * D * V'
% Sigma = V * D^2 * V' / N
% Sigma_pinv = V * D^(-2) * V'
Qs{3} = D\V'; % r x d
Qs{4} = ones(1,d); % 1 x d
%
disp('Computing taus...');
tic;
taus = cell(4,1);
N0 = 1000;
pi = randperm(N);
for i=1:4
    taus{i} = zeros(N0,1);
    for j=1:N0
        if mod(j,100) == 0
            fprintf(1, 'i=%d, j=%d\n', i, j);
        end
        taus{i}(j) = norm(Qs{i}*(A(pi(j),:)' - mu_all - yt(pi(j)) * mu), 2);
    end
end
disp('Done computing taus');
toc;
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
disp('Computing taus_agg...');
tic;
taus_agg = zeros(N0,1);
for i=1:4
  taus_agg = taus_agg + taus{i}.^2 / thresholds(i)^2;
end
taus_agg = sqrt(taus_agg);
%for j=1:N
%    if mod(j,100) == 0
%        fprintf(1, 'i=agg, j=%d\n', j);
%    end
%    taus_agg(j) = norm(Q_agg*(A(j,:)' - mu_all - yt(j) * mu), 2);
%end
toc;
disp('Saving results...');
tic;
save(sprintf('Q_agg_r%d.mat', r));
toc;
