data = load("Data.txt");
x = data(:,1);
y = data(:,2);

%plot(x,y,'xr')
m = length(y);
alpha = 0.01;
theta = zeros(2,1);
x = [ones(length(y),1),data(:,1)];
iterations = 1500;
%[x_norm, mu, sigma] = featuresNormalization(x)
theta = NormalEqu(x,y);
predvalue = x * theta;
plot(x,y,'xr')
hold on
plot(x,predvalue)
%[theta,cost_history] = gradientDescent(x, y, theta, alpha, iterations);
