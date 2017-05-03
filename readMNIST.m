function [X_train, y_train, X_test, y_test] = generate_mnist(num_classes)
    X_train = loadMNISTImages('mnist/train-images-idx3-ubyte');
    y_train = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
    X_train = X_train(:, y_train < num_classes)';
    y_train = y_train(y_train < num_classes) + 1;

    X_test = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
    y_test = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
    X_test = X_test(:, y_test < num_classes)';
    y_test = y_test(y_test < num_classes) + 1;
end
