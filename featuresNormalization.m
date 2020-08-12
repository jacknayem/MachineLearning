function [x_norm, mu, sigma] = featuresNormalization(x)
x_norm = x;
mu = zeros(1, size(x, 2));
sigma = zeros(1, size(x, 2));
mu = mean(x)
sigma = std(x)
x_norm = (x - mu)/sigma
end