clear;
%rng(400);
[Xtr,Ytr,ytr] = LoadBatch('data_batch_1.mat');
[Xva,Yva,yva] = LoadBatch('data_batch_2.mat');
[Xte,Yte,yte] = LoadBatch('test_batch.mat');
data.Xtr = Xtr;data.Ytr=Ytr;data.ytr=ytr;
data.Xva = Xva;data.Yva=Yva;data.yva=yva;
data.Xte = Xte;data.Yte=Yte;data.yte=yte;
d = size(Xtr,1);
N = size(Xtr,2);
K = size(Ytr,1);
lambda = .1;
GDparams = setGDparams(100,0.01,50);
% initialize parameters W b.
mean = 0; std = 0.01;
W = std.*randn(K,d)+mean;
b = std.*randn(K,1)+mean;
%[error_b,error_W] = gradientsCheck(Xtr, Ytr, W,b,50);
main(GDparams,data,W,b,lambda)
%% load function 
% X image pixel data, dxN, entries between 0-1. N is number of images
% d is dimensionality of each image (3072=32x32x3)
% Y is KxN, K is number of labels. we use one-hot representation here
% y is a vector of length N containing the label for each image.
% CIFAR-10 encodes 0-9, we use 1-10
function [X, Y, y] = LoadBatch(filename)
A = load(filename);
X = double(A.data')/255;
y = double(A.labels') + 1;
K = 10;
N = size(y,2);
Y = zeros(K,N);
for i=1:N
    Y(y(1,i),i) = 1;
end
end
%% network functions
function P = EvaluateClassifier(X,W,b)
k=size(W,1);
N = size(X,2);
b = repmat(b,1,N);
s = W*X + b;
P=exp(s)./sum(exp(s),1);
end
%% Cost function
function J = ComputeCost(X,Y,W,b,lambda)
N = size(X,2);
P = EvaluateClassifier(X,W,b);
cross = 0;
for i = 1:N
    cross = cross + (-log(Y(:,i)'*P(:,i)));
end
reg = sum(sum(W.^2));
J = cross/N + lambda*reg;
    
end
%% accuracy of the network's predictions
% becarefull that y here is the vector that contains the true lables.
function acc = ComputeAccuracy(X,y,W,b)
P = EvaluateClassifier(X,W,b);
[p,kstar] = max(P);
N=size(X,2);
acc = 0;
for i=1:N
    if kstar(i) == y(i)
        acc = acc + 1/N;
    end
end
end

%% gradients
function [grad_b, grad_W] = ComputeGradients(X, Y, P, W, lambda)
N = size(X,2);
k = size(W,1);
grad_W = zeros(size(W));
grad_b = zeros(k, 1);
for i=1:N
    g = -Y(:,i)'/(Y(:,i)'*P(:,i))*(diag(P(:,i))-P(:,i)*P(:,i)');
    grad_W = grad_W + g'*X(:,i)';
    grad_b = grad_b + g';
end
grad_W = grad_W/N + 2*lambda*W;
grad_b = grad_b/N;
end


%% gradients check
function [error_b, error_W] = gradientsCheck(X, Y, W,b, batch_size)
dim= size(X,1);
N = batch_size;
lambda=0;
h=1e-6;
P = EvaluateClassifier(X(1:dim,1:N),W,b);
[ngrad_b,ngrad_W] = ComputeGradsNumSlow(X(1:dim,1:N),Y(:,1:N),W,b,lambda,h);
[agrad_b,agrad_W] = ComputeGradients(X(1:dim,1:N),Y(:,1:N),P,W,lambda);

error_W = max(max(abs(ngrad_W-agrad_W)./max(h,abs(ngrad_W)+abs(agrad_W))))
error_b = max(abs(ngrad_b-agrad_b)./max(h,abs(ngrad_b)+abs(agrad_b)))
end

%% numeric gradients
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end
%% set GDparams
function GDparams = setGDparams(n_batch, eta, n_epochs)
if nargin > 0
    GDparams.n_batch = n_batch;
    GDparams.eta = eta;
    GDparams.n_epochs = n_epochs;
end
end

%% minibatchgd
function[Wstar,bstar] = MiniBatchGD(X,Y,GDparams,W,b,lambda)
n_batch = GDparams.n_batch;
eta = GDparams.eta;
N = size(X,2);

for j = 1 : N/n_batch
    j_start = (j-1)*n_batch +1;
    j_end = j*n_batch;
    inds = j_start : j_end;
    Xbatch = X(:,inds);
    Ybatch = Y(:,inds);
    
    P = EvaluateClassifier(Xbatch, W, b);
    [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch, P,W,lambda);
    
    W = W - eta*grad_W;
    b = b - eta*grad_b;
end

Wstar = W;
bstar = b;
end
%% main loop
function main(GDparams,data,W,b,lambda)
Jtr = zeros(1,GDparams.n_epochs);
Jva = zeros(1,GDparams.n_epochs);
for i = 1 : GDparams.n_epochs
    Jtr(i) = ComputeCost(data.Xtr,data.Ytr,W,b,lambda);
    Jva(i) = ComputeCost(data.Xva,data.Yva,W,b,lambda);
    [W, b] = MiniBatchGD(data.Xtr,data.Ytr,GDparams,W,b,lambda);
end
% accuracy
Acctr = ComputeAccuracy(data.Xtr,data.ytr,W,b);
disp(['training accuracy:' num2str(Acctr*100) '%'])
Accte = ComputeAccuracy(data.Xte,data.yte,W,b);
disp(['testing accuracy:' num2str(Accte*100) '%'])

% plots
figure()
plot(1:GDparams.n_epochs, Jtr, 'g')
hold on
plot(1:GDparams.n_epochs, Jva, 'r')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');
k = size(W,1);
for i= 1 : k
    im = reshape(W(i,:),32,32,3);
    s_im{i} = (im - min(im(:)))/(max(im(:))-min(im(:)));
    s_im{i} = permute(s_im{i},[2,1,3]);
end
figure()
montage(s_im,'size',[i,k])
end
