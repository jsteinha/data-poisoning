function showBow(x, y, vocab, cutoff)
    d = length(x);
    indices = [];
    if y == 1
        fprintf(1, '+:');
    else
        fprintf(1, '-:');
    end
    for i=1:d
        indices = [indices;i * ones(x(i),1)];
    end
    indices = indices(randperm(length(indices)));
    for i=indices(1:cutoff)
        fprintf(1, ' %s', vocab{i});
    end
    fprintf(1, ' ...\n');
end