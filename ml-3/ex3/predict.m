function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%disp(size(X)); 5000 x 401
%disp(size(Theta1)); 25 x 401
z1 = X * Theta1';
a1 = sigmoid(z1);
%disp(size(z)); 5000 x 25
%disp(size(a)); 5000 x 25

% Add ones to the X data matrix
a1 = [ones(m, 1) a1];

%disp(size(Theta2)); 10 x 26
z2 = a1 * Theta2';
a2 = sigmoid(z2);
%disp(size(z2)); % 5000 x 10
%disp(size(a2)); % 5000 x 10

[maxVal, maxIdx] = max(a2, [], 2);
J = [maxVal, maxIdx];
for c = 1:rows(maxVal)
    p(c) = J(c,2);
endfor

% =========================================================================


end
