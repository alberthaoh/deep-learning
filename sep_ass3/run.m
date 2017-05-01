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
