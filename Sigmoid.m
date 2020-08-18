function sigmoid = Sigmoid(z)
sigmoid = zeros(size(z));
sigmoid = 1 ./ (1 + exp(-z));
end