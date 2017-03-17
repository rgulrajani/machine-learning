function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Calculating the cost value without regularization.
costVal = (1/m) * sum(((-y)' * log(sigmoid(X * theta))) - ((1 - y)' * log(1 - sigmoid(X * theta))));
% Adding the regularized value only from the 2nd term to the end.
J = costVal + (lambda/(2*m)) * sum(theta(2:end) .^ 2);

% Calculating the gradient value without regularization.
gradVal = (1/m) * ((sigmoid(X * theta) - y)' * X);
% Calculating the regularized value.
regVal = (lambda / m) * theta;

grad = gradVal' + regVal;
% Setting the first value to the original value withoug regularization.
grad(1,1) = gradVal(1,1);

% =============================================================

end
