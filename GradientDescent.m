function [theta,hist] = GradientDescent(X,y,m,alpha, theta, iteration)
hist = zeros(m,1);
for i = 1:iteration
    z = X * theta;
    sigmoid = Sigmoid(z);
    theta = theta -((alpha/m) *(X' * (sigmoid - y)));
    hist(i) = CostFunction(X,y,theta,m);
end
end