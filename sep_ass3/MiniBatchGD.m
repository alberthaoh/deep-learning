%% minibatchgd
function[Wstar,bstar,muav,varsav] = MiniBatchGD(X,Y,Mparams,W,b,lambda)
n_batch = Mparams.n_batch;
N = size(X,2);
fieldW = fieldnames(W);
fieldb = fieldnames(b);
k = Mparams.n_layers;
alpha=0.9;
for j = 1:k
v_W.(fieldW{j}) = zeros(size(W.(fieldW{j})));
v_b.(fieldb{j}) = zeros(size(b.(fieldb{j})));
end
for j = 1 : N/n_batch
    j_start = (j-1)*n_batch +1;
    j_end = j*n_batch;
    inds = j_start : j_end;
    Xbatch = X(:,inds);
    Ybatch = Y(:,inds);
    [P,s,x,sNormed,mu,vars] = EvaluateClassifier(Xbatch, W, b,Mparams);
    %exponential moving average
    if j == 1
        muav=mu;
        varsav=vars;
    else
        muav=cellfun(@(x,y) alpha.*x+(1-alpha).*y,muav,mu,'un',0);
        varsav=cellfun(@(x,y) alpha.*x+(1-alpha).*y,varsav,vars,'un',0);
%         muav = alpha*muav+(1-alpha)*mu;
%         varsav = alpha*varsav+(1-alpha)*vars;
    end
    [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch,W,b,lambda,Mparams,P,s,x,sNormed,mu,vars);
 %   grad_W.One(1:10,1:10)
    Mparams.eta = Mparams.eta * Mparams.decay_rate;
    %Mparams.rho = Mparams.rho * (0.999);
    for i=1:k
        [v_W.(fieldW{i}),W.(fieldW{i})] = momentumUpdate(Mparams,v_W.(fieldW{i}),grad_W.(fieldW{i}),W.(fieldW{i}));
        [v_b.(fieldb{i}),b.(fieldb{i})] = momentumUpdate(Mparams,v_b.(fieldb{i}),grad_b.(fieldb{i}),b.(fieldb{i}));
    end
 %   W.One(1:10,1:10)
end

Wstar = W;
bstar = b;
end

