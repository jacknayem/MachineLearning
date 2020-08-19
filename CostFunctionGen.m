function J = CostFunctionGen(x,y,theta, lambda)
J = 0;
m = length(y);
error = (x * theta) - y;

gen = lambda/(2 * m) * sum(theta(2:end)^2);

J = 1/(2 * m) * sum(error)^2 + gen;
end