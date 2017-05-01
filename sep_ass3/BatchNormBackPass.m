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
