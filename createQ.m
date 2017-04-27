load imdb.mat;
path(pathdef);
%% construct Q matrix
Qs = cell(4,1);
Qs{1} = speye(d); % d x d
Qs{2} = mu'; % 1 x d
r = 2000;
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
