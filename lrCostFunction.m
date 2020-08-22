function [J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t)
J = 0;
grad = zeros(length(theta_t),1);
m = length(y_t);
z = X_t * theta_t;
hyp = sigmoid(z);
J = (1/m) .* (((-y_t)' * log(hyp))-((1 - y_t)' * log(1 - hyp))) + (lambda_t/(2 * m)) .* sum(theta_t(2:end).^2);
grad(1) = (1/m) .* (X_t(:,1)' * (hyp - y_t));
grad(2:end) = (1/m) .* (X_t(:,2:end)' * (hyp - y_t)) + (lambda_t/m) * theta_t(2:end);
end