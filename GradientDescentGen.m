function theta =  GradientDescentGen(x,y,theta,alpha, m, lambda, iterations)
for i = 1:iterations
temp1 = theta(1) - ((alpha/m) .* (x(:,1)' * ((x * theta) - y)));
temp2 = theta(2:end) - ((alpha/m) .* x(:,2:end)' * ((x * theta) - y)) + (lambda/m) * theta(2:end);
theta = [temp1;temp2];
end
end