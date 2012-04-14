function [val] = sigmoid(w, x)
	val = 1/(1 + exp(-(w*x')));
end
