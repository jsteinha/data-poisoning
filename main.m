clear;
f1 = fopen('train/labeledBow.feat');
N=25000;
M=1000000;
rs = zeros(M,1); cs = zeros(M,1); ss = zeros(M,1);
pos = 0;
y = zeros(N,1);
for i=1:N
    if(mod(i,100)==0)
        i
    end
    line = fgetl(f1);
    toks = sscanf(line, ['%d' repmat(' %d:%f', [1 sum(line==':')])]);
    c = toks(2:2:end)+1;
    s = toks(3:2:end);
    len = length(c);
    rs(pos+1:pos+len) = i;
    cs(pos+1:pos+len) = c;
    ss(pos+1:pos+len) = s;
    pos = pos + len;
    y(i) = toks(1);
end
rs = rs(1:pos); cs = cs(1:pos); ss = ss(1:pos);
A = sparse(rs, cs, ss);
yt = 2 * (logical(y>5)) - 1;
d = size(A, 2);
% test set
f1 = fopen('test/labeledBow.feat');
N=25000;
M=1000000;
rs = zeros(M,1); cs = zeros(M,1); ss = zeros(M,1);
pos = 0;
y_test = zeros(N,1);
for i=1:N
    if(mod(i,100) == 0)
        i
    end
    line = fgetl(f1);
    toks = sscanf(line, ['%d' repmat(' %d:%f', [1 sum(line==':')])]);
    c = toks(2:2:end)+1;
    s = toks(3:2:end);
    len = length(c);
    rs(pos+1:pos+len) = i;
    cs(pos+1:pos+len) = c;
    ss(pos+1:pos+len) = s;
    pos = pos + len;
    y_test(i) = toks(1);
end
rs = rs(1:pos); cs = cs(1:pos); ss = ss(1:pos);
A_test = sparse(rs, cs, ss, N, d);
yt_test = 2 * (logical(y_test>5)) - 1;
clear rs cs ss;
%% compute mu, sigma
mu_all = A' * ones(N,1) / N;
mu = A' * yt / N;
%A_n = diag(sparse(yt)) * A;
%[U, D, V] = svds(A_n / sqrt(N), 200);
%%
N_pert = round(0.10 * N);
y_pert = sign(randn(N_pert,1));
A_pert = sparse(N_pert,d);
B = 20;
for i=1:B:N_pert
    i
    i2 = min(i+B-1,N_pert);
    B2 = i2-i+1;
    A_pert_cur = ones(B2,1) * mu_all' + y_pert(i:i2) * xb';
    A_pert_round = round(A_pert_cur);
    A_pert_rem = A_pert_cur - A_pert_round;
    A_pert_rem_sparse = sign(A_pert_rem) .* (rand(B2,d) < abs(A_pert_rem));
    A_pert(i:i2,:) = A_pert_round + A_pert_rem_sparse;
end
clear A_pert_cur A_pert_round A_pert_rem A_pert_rem_sparse;
%% try to detect outliers
mu_poisoned = [A; A_pert]' * [yt; y_pert] / (N+N_pert);
mu_all_poisoned = [A; A_pert]' * ones(N+N_pert, 1) / (N + N_pert);
mu_plus = mu_all_poisoned + mu_poisoned;
mu_minus = mu_all_poisoned - mu_poisoned;
Q_sqrt = mu_poisoned;
%%
taus = zeros(N_pert,1);
tau = 0;
for i=1:N_pert
    if mod(i,100) == 0
        fprintf(1, 'iter %d: %.4f\n', i, tau / i);
    end
    if y_pert(i) == 1
        taus(i) = norm((A_pert(i,:) - mu_plus') * Q_sqrt, 2)^2;
    else
        taus(i) = norm((A_pert(i,:) - mu_minus') * Q_sqrt, 2)^2;
    end
    tau = tau + taus(i);
end
tau = tau / N_pert;
%%
tau0s = zeros(N,1);
tau0 = 0;
for i=1:N
    if mod(i,100) == 0
        fprintf(1, 'iter %d: %.4f\n', i, tau0/i);
    end
    if yt(i) == 1
        tau0s(i) = norm((A(i,:) - mu_plus') * Q_sqrt, 2)^2;
    else
        tau0s(i) = norm((A(i,:) - mu_minus') * Q_sqrt, 2)^2;
    end
    tau0 = tau0 + tau0s(i);
end
tau0 = tau0 / N;
%% train SVM
[lossTrain, accTrain, theta] = train(A, yt, 0.005, 1e-4, N, d);
% evaluation
[lossTest, accTest] = test(A_test, yt_test, N, d, theta);
%% poisoned data
v = A' * yt;
pv_1 = max(v,0); pv_1(1:200) = 0; ps_1 = sort(pv_1, 'descend'); pv_1 = pv_1 / ps_1(8);
pv_2 = max(-v,0); pv_2(1:150) = 0; ps_2 = sort(pv_2, 'descend'); pv_2 = pv_2 / ps_2(15);
%%
%Z = 35;
%pv_1 = max(v,0); ps_1 = sort(pv_1, 'descend'); t1 = ps_1(Z); pv_1h = pv_1 .* (pv_1 > t1) / max(pv_1); pv_1t = pv_1 .* (pv_1 <= t1) / t1;
%pv_2 = max(-v,0); ps_2 = sort(pv_2, 'descend'); t2 = ps_2(Z); pv_2h = pv_2 .* (pv_2 > t2) / max(pv_2); pv_2t = pv_2 .* (pv_2 <= t2) / t2;
N_pert = round(0.10 * N);
A_pert = sparse(N_pert,d);
B = 20;
for i=1:B:N_pert
    i
    i2 = min(i+B-1,N_pert);
    B2 = i2-i+1;
%     ok = 0;
%     while ~ok
%         if rand < 0.0
%             A_pert(i:i2,:) = A_pert(i:i2,:) + sparse(rand(B,d) < ones(B,1) * pv_1h');
%             ok=1;
%         end
%         if rand < 1.0
%             A_pert(i:i2,:) = A_pert(i:i2,:) + rand * sparse(rand(B,d) < ones(B,1) * pv_1t');
%             ok=1;
%         end
%         if rand < 0.0
%             A_pert(i:i2,:) = A_pert(i:i2,:) + sparse(rand(B,d) < ones(B,1) * pv_2h');
%             ok=1;
%         end
%         if rand < 1.0
%             A_pert(i:i2,:) = A_pert(i:i2,:) + rand * sparse(rand(B,d) < ones(B,1) * pv_2t');
%             ok=1;
%         end
%     end
        
    if rand < 0.7
        A_pert(i:i2,:) = A_pert(i:i2,:) + sparse(rand(B,d) < ones(B,1) * pv_1');
        if rand < 0.57
            A_pert(i:i2,:) = A_pert(i:i2,:) + sparse(rand(B,d) < ones(B,1) * pv_2');
        end
    else
        A_pert(i:i2,:) = A_pert(i:i2,:) + sparse(rand(B,d) < ones(B,1) * pv_2');
    end

%          A_pert(i:i2,:) = A_pert(i:i2,:) + rand * sparse(rand(B,d) < ones(B,1) * pv_1');
%          A_pert(i:i2,:) = A_pert(i:i2,:) + rand * sparse(rand(B,d) < ones(B,1) * pv_2');

end
%%
p = full(sum(sum(A))) / (N*d);
p_vec = full(sum(A,1)) / N;
mul = 4;
k = 20;
N_pert = round(0.30 * N);
U_pert = sparse(double(rand(N_pert,k) < mul/k));
V_pert = sparse(double(rand(k,d) < ones(k,1) * p_vec / mul));
A_pert = sparse(U_pert * V_pert);
%%
w_pert = -theta; %randn(d,1);
y_pert = sign(A_pert * w_pert);
%A_pert = A_pert(logical(y_pert~=0), :);
%y_pert = y_pert(logical(y_pert~=0));
%N_pert = size(A_pert, 1);
%%
[lossPert, accPert, thetaPert] = train([A;A_pert], [yt;y_pert], 0.005, 1e-4, N+N_pert, d);
[lossPertBad, accPertBad] = test(A_pert, y_pert, N_pert, d, thetaPert);
[lossPertTest, accPertTest] = test(A_test, yt_test, N, d, thetaPert);
%%
[lossPertRev, accPertRev] = test(A_pert, y_pert, N_pert, d, theta);
%%
fv = fopen('imdb.vocab');
vocab = cell(d,1);
for i=1:d
    word = fgetl(fv);
    vocab{i} = word;
end
%%
for i=1:20
    showBow(A_pert(i,:), y_pert(i), vocab, 10);
end
%%