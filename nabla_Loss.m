function [gradient, loss] = nabla_Loss(X, y, theta, verbose)
    N = size(X, 1);
    d = size(X, 2);
    gradient = zeros(d,1);
    loss = 0.0;
    for i=1:N
        margin = y(i) * X(i,:) * theta;
        loss = loss + max(1-margin, 0);
        if margin < 1
            gradient = gradient + y(i) * X(i,:)';
        end
        if verbose && mod(i,1000) == 0
            fprintf(1, '%d/%d examples...\n', i, N);
        end
    end
    loss = loss / N;
    gradient = -gradient / N; % reverse sign because we computed it wrong above
end
