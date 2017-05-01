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

