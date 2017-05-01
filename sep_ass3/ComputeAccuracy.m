%% accuracy of the network's predictions
% becarefull that y here is the vector that contains the true lables.
function acc = ComputeAccuracy(X,y,W,b,Mparams,muav,varsav)
P = EvaluateClassifier(X,W,b,Mparams,muav,varsav);
[p,kstar] = max(P);
N=size(X,2);
acc = 0;
for i=1:N
    if kstar(i) == y(i)
        acc = acc + 1/N;
    end
end
end
