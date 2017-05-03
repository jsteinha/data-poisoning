function [loss, acc, theta] = trainMulticlass(X, y, k, eta, delta, N, d, Nmax)
    theta = zeros(d,k);
    theta2 = delta * ones(d,k);
    totalLoss = zeros(k,1);
    totalAcc = zeros(k,1);
    counts = zeros(k,1);
    pi = randperm(N);
    for ii=1:min(N,Nmax)
        i = pi(ii);
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
        totalLoss(y(i)) = totalLoss(y(i)) + max(1-margin, 0); %0.5 * max(1-margin, 0)^2;
        totalAcc(y(i)) = totalAcc(y(i)) + (margin > 0);
        counts(y(i)) = counts(y(i)) + 1;
        if margin < 1
            gradient = X(i,:)'; % * (1-margin);
            theta2(:,y(i)) = theta2(:,y(i)) + gradient .* gradient;
            theta(:,y(i)) = theta(:,y(i)) + eta * gradient ./ sqrt(theta2(:,y(i)));
            theta2(:,bestj) = theta2(:,bestj) + gradient .* gradient;
            theta(:,bestj) = theta(:,bestj) - eta * gradient ./ sqrt(theta2(:,bestj));
        end
        if mod(ii,100) == 0
            fprintf(1, 'avg loss (iter %d): %.4f (%.4f)\n', ii, sum(totalLoss) / ii, sum(totalAcc) / ii);
        end
    end
    loss = totalLoss ./ counts; %min(N,Nmax);
    acc = totalAcc ./ counts; %min(N,Nmax);
end
