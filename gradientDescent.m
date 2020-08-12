function [theta,cost_history] = gradientDescent(x, y, theta, alpha, num_iters)
m = length(y)
cost_history = zeros(num_iters,1)
for iter = 1:num_iters
    error = (x * theta)-y;
    theta = theta - ((alpha/m) .* (x' * error));
    cost_history(iter) = CostFunction(x,y,theta,m)
end

end
