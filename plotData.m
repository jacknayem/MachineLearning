function value = plotData(x,y)
positive = find(y == 1);
negative = find(y ~= 1);
figure;
plot(x(positive,1),x(positive,2), 'k+');
hold on;
plot(x(negative,1),x(negative,2), 'ko');
hold off;
end