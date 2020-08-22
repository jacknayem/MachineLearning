function all_theta =  OneVsAll(X, y, number_label, lambda)
X = [ones(length(X),1), X];
[m,n] = size(X);
all_theta = zeros(number_label,n);

initial_theta = zeros(n,1);

options = optimset('GradObj','on','MaxIter',50);
for c = 1:number_label
    all_theta(c,:) = fminunc(@(t)(lrCostFunction(t, X, (y == c), lambda)),...
        initial_theta, options);
end
end