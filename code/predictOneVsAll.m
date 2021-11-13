function p = predictOneVsAll(all_theta, X)
m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];
pred = sigmoid( X * all_theta');

p = max(pred, [], 2);

for i = 1 : m
    for j = 1 : num_labels
        if( pred(i, j) == p(i, :))
            if( j ~= 10)
                p(i,:) = j;
            else
                p(i,:) = 0;
            end
        end
    end
end  

end
