%% derivative over Relu
function h_s = Ind(s)
h_s = diag(s>0);
end
