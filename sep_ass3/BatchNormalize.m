%% forward batchNormalize
function sNormed = BatchNormalize(sUnormed,mu,vars)
sNormed = (diag(vars+eps))^(-.5)*(sUnormed-mu);
end
