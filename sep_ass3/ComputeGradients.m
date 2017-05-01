%% gradients
function [grad_b, grad_W] = ComputeGradients(X, Y, W,b, lambda,Mparams,P,s,x,sNormed,mu,vars)
N = size(X,2);
fieldW = fieldnames(W);
fieldb = fieldnames(b);
k = Mparams.n_layers;
for j = 1:k
    grad_W.(fieldW{j}) = zeros(size(W.(fieldW{j})));
    grad_b.(fieldb{j}) = zeros(size(b.(fieldb{j})));
end
%g=zeros(N,10);
% size(s{1})
% s{1}(1:10,1)
% sNormed{1}(1:10,1)
for i=1:N
    g(i,:) = -Y(:,i)'/(Y(:,i)'*P(:,i))*(diag(P(:,i))-P(:,i)*P(:,i)');
    grad_W.(fieldW{k}) = grad_W.(fieldW{k}) + g(i,:)'*x{k}(:,i)';%x{1} is input
    grad_b.(fieldb{k}) = grad_b.(fieldb{k}) + g(i,:)';   
    g2(i,:)=g(i,:)*W.(fieldW{k})*Ind(sNormed{k-1}(:,i));
%     size(W.(fieldW{k}))
%     g(i,:) = g(i,:)*W.(fieldW{k});
%     g(i,:) = g(i,:)*Ind(sNormed{k-1}(:,i));
end
g=g2;
% g(1,1:10)
for j = k-1:-1:1
    g=BatchNormBackPass(g,s{j},mu{j},vars{j});
%     g(1,1:10)
    for i=1:N
        grad_W.(fieldW{j}) = grad_W.(fieldW{j}) + g(i,:)'*x{j}(:,i)';%x{1} is input
        grad_b.(fieldb{j}) = grad_b.(fieldb{j}) + g(i,:)';
        if j>1
            g3(i,:) = g(i,:)*W.(fieldW{j})*Ind(sNormed{j-1}(:,i));
        end
    end
    if Mparams.n_layers>2
        g =g3;
    end
end

for j = 1:k
    grad_W.(fieldW{j}) = grad_W.(fieldW{j})/N + 2*lambda*W.(fieldW{j});
    grad_b.(fieldb{j}) = grad_b.(fieldb{j})/N;
end
% W.One(1:10,1:10)
% grad_W.One(1:10,1:10)
end
