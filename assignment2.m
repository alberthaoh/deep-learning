clear;
rng(0);
% [Xtr,Ytr,ytr] = LoadBatch('data_batch_1.mat');
% [Xva,Yva,yva] = LoadBatch('data_batch_2.mat');
% [Xte,Yte,yte] = LoadBatch('test_batch.mat');

[Xtr1,Ytr1,ytr1] = LoadBatch('data_batch_1.mat');
[Xtr2,Ytr2,ytr2] = LoadBatch('data_batch_2.mat');
[Xtr3,Ytr3,ytr3] = LoadBatch('data_batch_3.mat');
[Xtr4,Ytr4,ytr4] = LoadBatch('data_batch_4.mat');
[Xtr5,Ytr5,ytr5] = LoadBatch('data_batch_5.mat');
Xtr=[Xtr1 Xtr2 Xtr3 Xtr4 Xtr5];
Ytr=[Ytr1 Ytr2 Ytr3 Ytr4 Ytr5];
ytr=[ytr1 ytr2 ytr3 ytr4 ytr5];
[Xva,Yva,yva] = LoadBatch('test_batch.mat');
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

lambda = 0.000005;
GDparams = setGDparams(900,.0702,30,0.9,0.99); %n_batch,eta,n_epochs,rho,decay_rate
%initialize parameters W b.
mean = 0; std = 0.01;
W.One = std.*randn(m,d)+mean;W.Two = std.*randn(K,m)+mean;
b.One = zeros(m,1); b.Two = zeros(K,1);
%[error_b,error_W] = gradientsCheck(Xtr, Ytr, W,b,5);
[result, Wed, bed] = main(GDparams,data,W,b,lambda)
% GDparams2 = setGDparams(500,0.102,10,0.9,0.975);
% main(GDparams2,data,Wed,bed,lambda)

% n = 200;
% result = zeros(n+1,8);
% for i = 1:n
%     lambda = 0.005 + (0.05-0.000001)*rand(1,1);
%     %lambda = 10^lam;
%     eta = 0.07+(0.1-0.07)*rand(1,1);
%     rho = 0.5+(1-0.5)*rand(1,1);
%     decay = 0.1 + (1-0.1)*rand(1,1);
%     %eta = 10^et;
%     %lambda = 0.00005;
%     %eta = 0.1+i*0.01;
%     result(1+i,2) = lambda;
%     result(1+i,3) = rho;
%     result(1+i,4) = decay;
%     result(1+i,1) = eta;
%     GDparams = setGDparams(100,eta,10,rho,decay);
%    % initialize parameters W b.
%     mean = 0; std = 0.001;
%     W.One = std.*randn(m,d)+mean;W.Two = std.*randn(K,m)+mean;
%     b.One = zeros(m,1); b.Two = zeros(K,1);
%     result(1+i,5:8)=main(GDparams,data,W,b,lambda);
%     i
%     if result(i+1,6) >= result(1,6)
%         result(1,:)=result(i+1,:);
%     end
% end

% figure()
% plot(result(:,1),result(:,4))
%% main loop
function [result,Wed,bed] = main(GDparams,data,W,b,lambda)
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
Wed = W;
bed = b;
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
[agrad_b,agrad_W] = ComputeGradients(X(1:dim,1:N),Y(:,1:N),W,b,lambda);

error_W.One = max(max(abs(ngrad_W.One-agrad_W.One)./max(epslo,abs(ngrad_W.One)+abs(agrad_W.One))))
error_b.One = max(abs(ngrad_b.One-agrad_b.One)./max(epslo,abs(ngrad_b.One)+abs(agrad_b.One)))
error_W.Two = max(max(abs(ngrad_W.Two-agrad_W.Two)./max(epslo,abs(ngrad_W.Two)+abs(agrad_W.Two))))
error_b.Two = max(abs(ngrad_b.Two-agrad_b.Two)./max(epslo,abs(ngrad_b.Two)+abs(agrad_b.Two)))
end

%% numeric gradients
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

%for j=1:length(b.One)
    grad_b.One = zeros(size(b.One));
    grad_b.Two = zeros(size(b.Two));
    for i=1:length(b.One)
        
        b_try = b;
        b_try.One(i) = b_try.One(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try.One(i) = b_try.One(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b.One(i) = (c2-c1) / (2*h);
    end
    for i=1:length(b.Two)
        
        b_try = b;
        b_try.Two(i) = b_try.Two(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try.Two(i) = b_try.Two(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b.Two(i) = (c2-c1) / (2*h);
    end

    grad_W.One = zeros(size(W.One));
    
    for i=1:numel(W.One)
        
        W_try = W;
        W_try.One(i) = W_try.One(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try.One(i) = W_try.One(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W.One(i) = (c2-c1) / (2*h);
    end
    grad_W.Two = zeros(size(W.Two));
    for i=1:numel(W.Two)
        
        W_try = W;
        W_try.Two(i) = W_try.Two(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try.Two(i) = W_try.Two(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W.Two(i) = (c2-c1) / (2*h);
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
    GDparams.rho = GDparams.rho * (0.999);
    [v_W.One,W.One] = momentumUpdate(GDparams,v_W.One,grad_W.One,W.One);
    [v_b.One,b.One] = momentumUpdate(GDparams,v_b.One,grad_b.One,b.One);
    [v_W.Two,W.Two] = momentumUpdate(GDparams,v_W.Two,grad_W.Two,W.Two);
    [v_b.Two,b.Two] = momentumUpdate(GDparams,v_b.Two,grad_b.Two,b.Two);
end

Wstar = W;
bstar = b;
end

