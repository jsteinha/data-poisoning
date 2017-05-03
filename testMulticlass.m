function [loss, acc] = testMulticlass(X, y, k, N, d, theta, Nmax)
    totalLossTest = zeros(k,1);
    totalAccTest = zeros(k,1);
    counts = zeros(k,1);
    piTest = randperm(N);
    for ii=1:min(N,Nmax)
        i = piTest(ii);
        %margin = y(i) * X(i,:) * theta;
        plus = X(i,:) * theta(:, y(i));
        minus = -99999;
        bestj = -1;
        for j=1:k
          if j ~= y(i)
            curminus = X(i,:) * theta(:,j);
            if bestj == -1 || curminus > minus
              bestj = j;
              minus = curminus;
            end
          end
        end
        margin = plus - minus;
        totalLossTest(y(i)) = totalLossTest(y(i)) + max(1-margin, 0); %0.5 * max(1-margin,0)^2;
        totalAccTest(y(i)) = totalAccTest(y(i)) + (margin > 0);
        counts(y(i)) = counts(y(i)) + 1;
        if mod(ii,100) == 0
            fprintf(1, 'avg loss (iter %d): %.4f (%.4f)\n', ii, sum(totalLossTest) / ii, sum(totalAccTest) / ii);
        end
    end
    loss = totalLossTest ./ counts; %min(N,Nmax);
    acc = totalAccTest ./ counts; %min(N,Nmax);
end
