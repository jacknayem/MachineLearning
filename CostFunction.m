function [J,grad] = CostFunction(X,y,theta,m)
J = 0;
z = X * theta;
hpyX = Sigmoid(z);
J = (1/m) .* ((-y' * log(hpyX)) - ((1-y)' * log(1-hpyX)));
grad = (1/m) * (X' * (hpyX - y));
end