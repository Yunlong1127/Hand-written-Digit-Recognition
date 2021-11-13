function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y); 
J = 0;
grad = zeros(size(theta));
[row, col] = size(X);


cost = (-1 .* y)' * log(sigmoid( X * theta))-(ones(row,1) - y)' * log(ones(row,1) - sigmoid(X * theta));

thet = theta(2:col,1);

penal =  lambda / 2 * ((norm(thet))^2);

J = 1 / m * (cost + penal);
tmp = X' * (sigmoid(X* theta) - y);

theta(1,1) = 0; 
pena = lambda .* theta;
grad = tmp + pena;
grad = (1 / m) .* grad(:);
    
end

