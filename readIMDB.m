f1 = fopen('imdb/train/labeledBow.feat');
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
f1 = fopen('imdb/test/labeledBow.feat');
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
% compute mu, sigma
mu_all = A' * ones(N,1) / N;
mu = A' * yt / N;
save imdb.mat A yt A_test yt_test mu_all mu N d;
