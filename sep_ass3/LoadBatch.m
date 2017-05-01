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
