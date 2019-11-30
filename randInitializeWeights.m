function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first row of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out,1 + L_in);
range = 0.12;
% Multipy by two to get both range [positve, negative]
W = 2* rand(L_out, 1 + L_in)* range-range;

% =========================================================================

end
