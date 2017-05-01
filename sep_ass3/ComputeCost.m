%% Cost function
function J = ComputeCost(X,Y,W,b,lambda,Mparams,muav,varsav)
N = size(X,2);
[P,s,x,sNormed,mu,vars] = EvaluateClassifier(X,W,b,Mparams,muav,varsav);
cross = 0;
for i = 1:N
    cross = cross + (-log(Y(:,i)'*P(:,i)));
end
reg = sum(sum(W.One.^2)) + sum(sum(W.Two.^2));
J = cross/N + lambda*reg;
    
end
