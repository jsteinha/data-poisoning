function [loss, acc] = test(X, y, N, d, theta)
    totalLossTest = 0;
    totalAccTest = 0;
    piTest = randperm(N);
    for ii=1:N
        i = piTest(ii);
        margin = y(i) * X(i,:) * theta;
        totalLossTest = totalLossTest + max(1-margin, 0); %0.5 * max(1-margin,0)^2;
        totalAccTest = totalAccTest + (margin > 0);
        if mod(ii,100) == 0
            fprintf(1, 'avg loss (iter %d): %.4f (%.4f)\n', ii, totalLossTest / ii, totalAccTest / ii);
        end
    end
    loss = totalLossTest / N;
    acc = totalAccTest / N;
end