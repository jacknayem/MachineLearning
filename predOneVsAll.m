function pred = predOneVsAll(all_theta,X)
m = size(X,1);
X = [ones(m,1) X];
pred = zeros(m,1);
prob = X * all_theta';
[prob, pred] = max(prob,[],2);
end