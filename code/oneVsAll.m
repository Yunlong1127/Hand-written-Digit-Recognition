function [all_theta] = oneVsAll(X, y, num_labels, lambda)
m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);
X = [ones(m, 1) X];
for i = 1 : num_labels
    y_tmp = (y == i);
    initial_theta = zeros(n + 1, 1);
    
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [all_theta(i,:)] = fmincg (@(t)(lrCostFunction(t, X, y_tmp, lambda)),initial_theta, options);
end

end
