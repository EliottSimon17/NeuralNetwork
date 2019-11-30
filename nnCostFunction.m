function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);


% Computation of the Cost function including regularisation
% Feedforward 
% We can deduce here that a1 = ones[m, 1]
a1 = [ones(m, 1) X];

a2 = sigmoid([ones(m, 1) X] * Theta1');
step2 = [ones(m, 1) a2];
a3 = sigmoid([ones(m, 1) a2] * Theta2');

% Cost function for Logistic Regression summed over all output nodes
for k=1:num_labels
    % which examples fit this label
    y_binary=(y==k);
    % select all predictions for label k
    hk=a3(:,k);
    % compute two parts of cost function for all examples for node k
    Cost(k,:) = y_binary'*log(hk) + ((1-y_binary')*log(1-hk));  
end
% Sum over all labels and average over examples
J_no_regularisation = -1./m * sum(Cost);
% No regularization over intercept
Theta1_no_intercept = Theta1(:,2:end);
Theta2_no_intercept = Theta2(:,2:end);
% Sum all parameters squared
RegSum1 = sum(sum(Theta1_no_intercept.^2));
RegSum2 = sum(sum(Theta2_no_intercept.^2));
% Add regularisation term to final cost
J = J_no_regularisation + (lambda./(2*m)) * (RegSum1+RegSum2);

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Implement the backpropagation algorithm to compute the gradients
% Theta1_grad and Theta2_grad. You should return the partial derivatives of
% the cost function with respect to Theta1 and Theta2 in Theta1_grad and
% Theta2_grad, respectively. After implementing Part 2, you can check
% that your implementation is correct by running checkNNGradients
%
% Note: The vector y passed into the function is a vector of labels
%       containing values from 1..K. You need to map this vector into a 
%       binary vector of 1's and 0's to be used with the neural network
%       cost function.
%
% Hint: It is recommended implementing backpropagation using a for-loop
%       over the training examples if you are implementing it for the 
%       first time.
%
for t = 1:m
    for k = 1:num_labels
        yk = y(t) == k;
        delta_3(k) = a3(t, k) - yk;
        
       
    end
    a2(t,:)
    [1, a2(t, :)]
    delta_2 = Theta2' * delta_3' .* (sigmoidGradient([1, a2(t, :)]))';
    delta_2 = delta_2(2:end);

    Theta1_grad = Theta1_grad + delta_2 * a1(t, :);
    Theta2_grad = Theta2_grad + delta_3' * step2(t, :);
end

%Take the average of all the theta gradients we computed earlier
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

%regulariztion
%Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
%Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients 
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
