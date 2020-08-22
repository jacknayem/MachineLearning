function hyp = sigmoid(z)
hyp = 1 ./ (1 + exp(-z));
end