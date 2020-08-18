function [J,grad] = Regularization(X, y, theta, labda,m)
J = 0;
grad = zeros(size(theta));
z = X * theta
hyp = Sigmoid(z)

J = (1/m).* ((-y' * log(hyp)) - ((1-y)' * log(1 - hyp)))  + ((labda ./ (2 .* m)) .* sum(theta(2:end)));

grad(1) = (1/m) .* (X(:,1)' * (hyp - y));
grad(2:end) = (1/m) * (X(:,2:end)' * (hyp - y)) + (labda/m)* theta(2:end)
end