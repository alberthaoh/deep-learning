%% momentum update
function [vNew,parasNew] = momentumUpdate(Mparams,vOld,gradient,paras)
    vNew = Mparams.rho.*vOld + Mparams.eta.*gradient;
    parasNew = paras - vNew;
end
