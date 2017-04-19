clear;
rng(400);
[Xtr,Ytr,ytr] = LoadBatch('data_batch_1.mat');
[Xva,Yva,yva] = LoadBatch('data_batch_2.mat');
[Xte,Yte,yte] = LoadBatch('test_batch.mat');
mean_Xtr = mean(Xtr,2);
Xtr = Xtr - repmat(mean_Xtr,[1,size(Xtr,2)]);
Xva = Xva - repmat(mean_Xtr,[1,size(Xva,2)]);
Xte = Xte - repmat(mean_Xtr,[1,size(Xte,2)]);
data.Xtr = Xtr;data.Ytr=Ytr;data.ytr=ytr;
data.Xva = Xva;data.Yva=Yva;data.yva=yva;
data.Xte = Xte;data.Yte=Yte;data.yte=yte;
data.Xtr = Xtr;data.Ytr=Ytr;data.ytr=ytr;
data.Xva = Xva;data.Yva=Yva;data.yva=yva;
data.Xte = Xte;data.Yte=Yte;data.yte=yte;
d = size(Xtr,1);
N = size(Xtr,2);
K = size(Ytr,1);
m = 50;

lambda = 0.005;
GDparams = setGDparams(100,0.23,30,0.3,0.99); %n_batch,eta,n_epochs,rho,decay_rate
%initialize parameters W b.
mean = 0; std = 0.001;
W.One = std.*randn(m,d)+mean;W.Two = std.*randn(K,m)+mean;
b.One = zeros(m,1); b.Two = zeros(K,1);
%[error_b,error_W] = gradientsCheck(Xtr, Ytr, W,b,50);
main(GDparams,data,W,b,lambda)

% n = 10;
% result = zeros(n+1,6);
% for i = 1:n
%    % rho = -6 + (1+6)*rand(1,1);
%    % rho = 10^e;
%     rho = rand(1,1);
%     result(1+i,1) = rho;
%     de = rand(1,1);
%     lambda = 0.005;
%     result(1+i,2) = 0.995;
%     GDparams = setGDparams(100,0.023,3,rho,0.995);
%    % initialize parameters W b.
%     mean = 0; std = 0.001;
%     W.One = std.*randn(m,d)+mean;W.Two = std.*randn(K,m)+mean;
%     b.One = zeros(m,1); b.Two = zeros(K,1);
%     result(1+i,3:6)=main(GDparams,data,W,b,lambda);
%     i
%     if result(i+1,4) >= result(1,4)
%         result(1,:)=result(i+1,:);
%     end
% end
%figure()
%plot(result(:,1),result(:,5))
%% main loop
function result = main(GDparams,data,W,b,lambda)
Jtr = zeros(1,GDparams.n_epochs);
Jva = zeros(1,GDparams.n_epochs);
for i = 1 : GDparams.n_epochs
%     if (i==5)
%         GDparams.rho = 0.9;
%         GDparams.eta = 0.01;
%         GDparams.decay_rate=0.95;
%     end
    Jtr(i) = ComputeCost(data.Xtr,data.Ytr,W,b,lambda);
    Jva(i) = ComputeCost(data.Xva,data.Yva,W,b,lambda);
    [W, b] = MiniBatchGD(data.Xtr,data.Ytr,GDparams,W,b,lambda);
end
% accuracy
Acctr = ComputeAccuracy(data.Xtr,data.ytr,W,b);
disp(['training accuracy:' num2str(Acctr*100) '%'])
Accte = ComputeAccuracy(data.Xte,data.yte,W,b);
disp(['testing accuracy:' num2str(Accte*100) '%'])
result = [Acctr Accte Jtr(GDparams.n_epochs) Jva(GDparams.n_epochs)];

%plots
figure()
plot(1:GDparams.n_epochs, Jtr, 'g')
hold on
plot(1:GDparams.n_epochs, Jva, 'r')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');

%k = size(W,1);
%for i= 1 : k
%    im = reshape(W(i,:),32,32,3);
%    s_im{i} = (im - min(im(:)))/(max(im(:))-min(im(:)));
%    s_im{i} = permute(s_im{i},[2,1,3]);
%end
%figure()
%montage(s_im,'size',[i,k])
end
%% momentum update
function [vNew,parasNew] = momentumUpdate(GDparams,vOld,gradient,paras)
    vNew = GDparams.rho.*vOld + GDparams.eta.*gradient;
    parasNew = paras - vNew;
end
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
%X = X(:,1:1000);
%Y = Y(:,1:1000);
%y = y(:,1:1000);
end
%% network functions
function [P,h,s] = EvaluateClassifier(X,W,b)
%k=size(W,1);
N = size(X,2);
% first layer
b.One = repmat(b.One,1,N);b.Two = repmat(b.Two,1,N);
s.One = W.One*X + b.One;
h=max(0,s.One);
% second layer
s.Two = W.Two*h + b.Two;
P=exp(s.Two)./sum(exp(s.Two),1);
end
%% Cost function
function J = ComputeCost(X,Y,W,b,lambda)
N = size(X,2);
[P,h,s] = EvaluateClassifier(X,W,b);
cross = 0;
for i = 1:N
    cross = cross + (-log(Y(:,i)'*P(:,i)));
end
reg = sum(sum(W.One.^2)) + sum(sum(W.Two.^2));
J = cross/N + lambda*reg;
    
end
%% accuracy of the network's predictions
% becarefull that y here is the vector that contains the true lables.
function acc = ComputeAccuracy(X,y,W,b)
[P,h,s] = EvaluateClassifier(X,W,b);
[p,kstar] = max(P);
N=size(X,2);
acc = 0;
for i=1:N
    if kstar(i) == y(i)
        acc = acc + 1/N;
    end
end
end
%% derivative over Relu
function h_s = Ind(s)
h_s = diag(s>0);
end
%% gradients
function [grad_b, grad_W] = ComputeGradients(X, Y, W,b, lambda)
N = size(X,2);
grad_W.One = zeros(size(W.One));grad_W.Two = zeros(size(W.Two));
grad_b.One = zeros(size(b.One));grad_b.Two = zeros(size(b.Two));
[P,h,s] = EvaluateClassifier(X, W, b);
for i=1:N
    g = -Y(:,i)'/(Y(:,i)'*P(:,i))*(diag(P(:,i))-P(:,i)*P(:,i)');
    grad_W.Two = grad_W.Two + g'*h(:,i)';
    grad_b.Two = grad_b.Two + g';
    g2 = g*W.Two;
    g2 = g2*Ind(s.One(:,i));
    grad_W.One = grad_W.One + g2'*X(:,i)';
    grad_b.One = grad_b.One + g2';
    
end
grad_W.One = grad_W.One/N + 2*lambda*W.One;
grad_W.Two = grad_W.Two/N + 2*lambda*W.Two;
grad_b.One = grad_b.One/N;
grad_b.Two = grad_b.Two/N;
end


%% gradients check
function [error_b, error_W] = gradientsCheck(X, Y, W,b, batch_size)
dim= size(X,1);
N = batch_size;
lambda=0;
epslo=1e-5;
P = EvaluateClassifier(X(1:dim,1:N),W,b);
[ngrad_b,ngrad_W] = ComputeGradsNumSlow(X(1:dim,1:N),Y(:,1:N),W,b,lambda,epslo);
[agrad_b,agrad_W] = ComputeGradients(X(1:dim,1:N),Y(:,1:N),P,W,h,b,lambda);

error_W = max(max(abs(ngrad_W-agrad_W)./max(epslo,abs(ngrad_W)+abs(agrad_W))))
error_b = max(abs(ngrad_b-agrad_b)./max(epslo,abs(ngrad_b)+abs(agrad_b)))
end

%% numeric gradients
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end
%% set GDparams
function GDparams = setGDparams(n_batch, eta, n_epochs,rho,decay_rate)
if nargin > 0
    GDparams.n_batch = n_batch;
    GDparams.eta = eta;
    GDparams.n_epochs = n_epochs;
    GDparams.rho = rho;
    GDparams.decay_rate = decay_rate;
end
end

%% minibatchgd
function[Wstar,bstar] = MiniBatchGD(X,Y,GDparams,W,b,lambda)
n_batch = GDparams.n_batch;
eta = GDparams.eta;
decay_rate = GDparams.decay_rate;
N = size(X,2);
v_W.One= zeros(size(W.One));v_W.Two= zeros(size(W.Two));
v_b.One= zeros(size(b.One));v_b.Two= zeros(size(b.Two));
for j = 1 : N/n_batch
    j_start = (j-1)*n_batch +1;
    j_end = j*n_batch;
    inds = j_start : j_end;
    Xbatch = X(:,inds);
    Ybatch = Y(:,inds);
    [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch,W,b,lambda);
    GDparams.eta = GDparams.eta * GDparams.decay_rate;
 %   GDparams.rho = GDparams.rho * (GDparams.decay_rate+0.04);
    [v_W.One,W.One] = momentumUpdate(GDparams,v_W.One,grad_W.One,W.One);
    [v_b.One,b.One] = momentumUpdate(GDparams,v_b.One,grad_b.One,b.One);
    [v_W.Two,W.Two] = momentumUpdate(GDparams,v_W.Two,grad_W.Two,W.Two);
    [v_b.Two,b.Two] = momentumUpdate(GDparams,v_b.Two,grad_b.Two,b.Two);
end

Wstar = W;
bstar = b;
end

