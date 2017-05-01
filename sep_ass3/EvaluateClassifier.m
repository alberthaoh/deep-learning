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
