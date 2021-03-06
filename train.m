function [loss, acc, theta] = train(X, y, eta, delta, N, d, Nmax, numIters, lambda, verbose)
    theta = zeros(d,1);
    theta2 = delta * ones(d,1);
    for iter=1:numIters
        totalLoss = 0;
        totalAcc = 0;
        pi = randperm(N);
        for ii=1:min(N,Nmax)
            i = pi(ii);
            margin = y(i) * X(i,:) * theta;
            totalLoss = totalLoss + max(1-margin, 0); %0.5 * max(1-margin, 0)^2;
            totalAcc = totalAcc + (margin > 0);
            gradient = -lambda * theta;
            if margin < 1
                gradient = y(i) * X(i,:)'; % * (1-margin);
            end
            theta2 = theta2 + gradient .* gradient;
            theta = theta + eta * gradient ./ sqrt(theta2);
            if verbose && mod(ii,100) == 0
                fprintf(1, 'avg loss (iter %d): %.4f (%.4f)\n', ii, totalLoss / ii + 0.5 * lambda * norm(theta, 2)^2, totalAcc / ii);
            end
        end
        loss = totalLoss / min(N,Nmax);
        acc = totalAcc / min(N,Nmax);
    end
    loss = loss + 0.5 * lambda * norm(theta, 2)^2;
end
