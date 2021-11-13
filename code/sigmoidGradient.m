function g = sigmoidGradient(z)
g = zeros(size(z));

[rows cols] = size(g);

if cols == 1 && rows == 1
    g = sigmoid(z) * (ones(size(g)) - sigmoid(z));
else
    g = sigmoid(z) .* (ones(size(g)) - sigmoid(z));
end
