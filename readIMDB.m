f1 = fopen('imdb/train/labeledBow.feat');
N=25000;
M=1000000;
rs = zeros(M,1); cs = zeros(M,1); ss = zeros(M,1);
pos = 0;
y_train = zeros(N,1);
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
    y_train(i) = toks(1);
end
rs = rs(1:pos); cs = cs(1:pos); ss = ss(1:pos);
X_train = sparse(rs, cs, ss);
y_train = 2 * (logical(y_train>5)) - 1;
d = size(X_train, 2);
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
X_test = sparse(rs, cs, ss, N, d);
y_test = 2 * (logical(y_test>5)) - 1;
clear line toks len pos M f1 i c s rs cs ss;
