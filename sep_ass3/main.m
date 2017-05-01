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
