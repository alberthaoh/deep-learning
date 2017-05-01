clear;
% [Xtr,Ytr,ytr] = LoadBatch('data_batch_1.mat');
% [Xva,Yva,yva] = LoadBatch('data_batch_2.mat');
% [Xte,Yte,yte] = LoadBatch('test_batch.mat');

% Xtr=Xtr(1:10,1:10);Xva=Xva(1:10,1:10);
% Ytr=Ytr(1:10,1:10);Yva=Yva(1:10,1:10);
% ytr=ytr(1,1:10);yva=yva(1,1:10);
% Xte=Xte(1:10,1:10);Yte=Yte(1:10,1:10);
% yte=yte(1,1:10);


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


lambda = 0.000005;
[Mparams,W,b] = setMparams(1000,.15,30,0.9,0.975,data,[3,50,30]); %n_batch,eta,n_epochs,rho,decay_rate
main(Mparams,data,W,b,lambda)
%[error_b,error_W] = gradientsCheck(Xtr, Ytr, W,b,5);

 %[result, Wed, bed] = main(Mparams,data,W,b,lambda)
% Mparams2 = setMparams(500,0.102,10,0.9,0.975);
% main(Mparams2,data,Wed,bed,lambda)

% n = 10;
% result = zeros(n+1,8);
% for i = 1:n
%     rng(i);
% %     lambda = 0.005 + (0.05-0.000001)*rand(1,1);
%     lambda = 0.000005;
%     eta = 0.1+(0.8-0.1)*rand(1,1);
% %     rho = 0.5+(1-0.5)*rand(1,1);
% %     decay = 0.1 + (1-0.1)*rand(1,1);
%     rho = 0.9;
%     decay=0.975;
%     %eta = 10^et;
%     %lambda = 0.00005;
%     %eta = 0.1+i*0.01;
%     result(1+i,2) = lambda;
%     result(1+i,3) = rho;
%     result(1+i,4) = decay;
%     result(1+i,1) = eta;
%     [Mparams,W,b] = setMparams(100,eta,5,rho,decay,data,[3,50,30]);
%    % initialize parameters W b.
% %     mean = 0; std = 0.001;
% %     W.One = std.*randn(m,d)+mean;W.Two = std.*randn(K,m)+mean;
% %     b.One = zeros(m,1); b.Two = zeros(K,1);
%     result(1+i,5:8)=main(Mparams,data,W,b,lambda);
%     i
%     if result(i+1,6) >= result(1,6)
%         result(1,:)=result(i+1,:);
%     end
% end

% figure()
% plot(result(:,1),result(:,4))
%% main loop
function [result,Wed,bed] = main(Mparams,data,W,b,lambda)
Jtr = zeros(1,Mparams.n_epochs);
Jva = zeros(1,Mparams.n_epochs);
for i = 1 : Mparams.n_epochs
%     if (i==5)
%         Mparams.rho = 0.9;
%         Mparams.eta = 0.01;
%         Mparams.decay_rate=0.95;
%     end
    [W, b,muav,varsav] = MiniBatchGD(data.Xtr,data.Ytr,Mparams,W,b,lambda);
    Jtr(i) = ComputeCost(data.Xtr,data.Ytr,W,b,lambda,Mparams,muav,varsav);
    Jva(i) = ComputeCost(data.Xva,data.Yva,W,b,lambda,Mparams,muav,varsav);
end
% accuracy
Acctr = ComputeAccuracy(data.Xtr,data.ytr,W,b,Mparams,muav,varsav);
disp(['training accuracy:' num2str(Acctr*100) '%'])
Accte = ComputeAccuracy(data.Xte,data.yte,W,b,Mparams,muav,varsav);
disp(['testing accuracy:' num2str(Accte*100) '%'])
result = [Acctr Accte Jtr(Mparams.n_epochs) Jva(Mparams.n_epochs)];
Wed = W;
bed = b;
%plots
figure()
plot(1:Mparams.n_epochs, Jtr, 'g')
hold on
plot(1:Mparams.n_epochs, Jva, 'r')
dim = [0.2 0.5 0.3 0.3];
str = {'eta:',Mparams.eta,Accte};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
hold off
%plot(1:Mparams.n_epochs,means)
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
%% forward batchNormalize
function sNormed = BatchNormalize(sUnormed,mu,vars)
sNormed = (diag(vars+eps))^(-.5)*(sUnormed-mu);
end
%% backward batchNormalize
function gNew = BatchNormBackPass(gOld,s,mu,vars)
Vb = diag(vars+eps);
gNew = zeros(size(gOld));
J_var = 0;
J_mu = 0;
N = size(gOld,1);
for i=1:N
    J_var = J_var - gOld(i,:)*Vb^(-1.5)*diag(s(:,i)-mu);
    J_mu = J_mu - gOld(i,:)*Vb^(-.5);
end
%     size(gOld(i,:))
%     size(Vb)
%     size(diag(s(:,i)-mu))
% size(J_var)
% size(Vb)
for i = 1:N
    gNew(i,:) = gOld(i,:)*Vb^(-.5)+J_var*diag(s(:,i)-mu)/N+J_mu/N;
end
end
%% momentum update
function [vNew,parasNew] = momentumUpdate(Mparams,vOld,gradient,paras)
    vNew = Mparams.rho.*vOld + Mparams.eta.*gradient;
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
function [P,s,x,sNormed,mu,vars] = EvaluateClassifier(X,W,b,Mparams,varargin)
x{1}=X;
N = size(X,2);
fieldW = fieldnames(W);
fieldb = fieldnames(b);
k = Mparams.n_layers;
    for i = 1:k-1
        b.(fieldb{i})=repmat(b.(fieldb{i}),1,N);
        s{i} = W.(fieldW{i})*x{i} + b.(fieldb{i});
        switch nargin-4
            case 0
                mu{i} = mean(s{i},2);
                vars{i} = var(s{i},0,2)*(N-1)/N;
            case 2
                mu{i}=varargin{1}{i};
                vars{i}=varargin{2}{i};
            otherwise
                error('unexpected inputs')
        end
        sNormed{i} = BatchNormalize(s{i},mu{i},vars{i}); 
        x{i+1} = max(0,sNormed{i});
    end
s{k} = W.(fieldW{k})*x{k} + b.(fieldb{k});
P = exp(s{k})./sum(exp(s{k}),1);
end
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
%% derivative over Relu
function h_s = Ind(s)
h_s = diag(s>0);
end
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
%% set Mparams
function [Mparams,W,b] = setMparams(n_batch, eta, n_epochs,rho,decay_rate,data,layers)
if nargin > 0
    Mparams.n_batch = n_batch;
    Mparams.eta = eta;
    Mparams.n_epochs = n_epochs;
    Mparams.rho = rho;
    Mparams.decay_rate = decay_rate;
    Mparams.n_layers = layers(1);
end
rng(0);
d = size(data.Xtr,1);
K = size(data.Ytr,1);
mean = 0; std = 0.001;
if layers(1)==2
    W.One = std.*randn(layers(2),d)+mean;W.Two = std.*randn(K,layers(2))+mean;
    b.One = zeros(layers(2),1); b.Two = zeros(K,1);
end
if layers(1)==3
    W.One = std.*randn(layers(2),d)+mean;W.Two = std.*randn(layers(3),layers(2))+mean;
    b.One = zeros(layers(2),1); b.Two = zeros(layers(3),1);
    W.Three = std.*randn(K,layers(3))+mean;b.Three = zeros(K,1);
end
end

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

