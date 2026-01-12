clear
clc

%% 数值模拟

% paremters
addpath('dataset')
% % L: detection location
L = 5.1;
% % boundary_type，=1瞬时注入，=2连续, =3脉冲
% boundary_type = 1;
% % n=1，initial concentration,
n = 1;
%%%%%%%% data input
%%%% the memory expression Ks has been transformed by time-shift property
%%%% of Laplace transform
syms s
if L == 1.4
    load('river014.mat')
%     Ks = (log(s + 0.5835) * 5.8622) + -1.5738;  Ks = Ks + 42*s/L; %Ks = Ks -  log(sum(y))/L; l2_loss
    fs = 1.9492774/(s + 0.35011524)^5.910334; Ks = -1/L*log(fs); Ks = Ks + 42*s/L; %/20250814_172657_YmPzMV
%     Ks = 30*s+4.2217*log(s+0.35011524)-0.4769; % 20250814_172657_YmPzMV 
%     Ks = 11.616531 - 12.584847/(s + 1.0924276); Ks = Ks - log(sum(y))/L; Ks = Ks + 42*s/L ; %/20250810_140027_jQ1bOx
elseif L == 3.1
    load('river031.mat')
%     fs = 1.57208/((s + 0.08277482)^2.5922902);Ks = -1/L*log(fs); Ks = Ks + 90*s/L; %20250815_144948_pHS0Pi
%     fs =  (0.9654059/(s + 0.11159987))^3.1821866; Ks = -1/L*log(fs); Ks = Ks + 90*s/L; % \20250815_170117_jwFhjk
    Ks = log(s + 0.10748774) + 0.013656981; Ks = Ks + 90*s/L; %20250917_211023_cKBbhY
%     Ks = -0.5358925/(s + 0.24222377);     Ks = Ks + 90*s/L ; % no normalization
elseif L == 5.1
    load('river051.mat')
%     Ks = log(2.1378236*s + 0.26527378);Ks = Ks + 130*s/L; %20251209_162733_QQ3LyB
%     fs = (34.129555/(s*67.502014 + 5.513775))^3.7501204; Ks = -1/L*log(fs); Ks = Ks + 130*s/L; % ks= 3.753034 * log(1.9778819 * s + 0.161818774)/L
%     Ks = 0.7353*(log(s+0.0817)+34.66*s+0.681); %同↑表达式34.12855.../20250809_153738_omsiS5
    Ks = 0.19638655*log(s) + 0.17883855 - 0.0131630255/s;Ks = Ks + 130*s/L; 
else
    warning('This is not for CTRW!')
    load('saa50.mat')
    Ks = 0.6162712/(s^(-0.88989747) + 0.8696689);
    L = 50;
end


% y = rFMIM(t,alpha, beta, mu,v,d,x,boundary_type, pulse_time,n);%调参关键FHTE(time,\alpha,velocity,Dispersion coef (D),outflow location (x))
% ft = temper_FADE(t,alpha, mu,v,d,x,boundary_type, pulse_time,n); %(t,alpha, mu,v,d,x,boundary,pulse_time, n);y_noise = y.*(1+0.05*(2*rand(size(y))-1));
% ft = sr_FADE(t,Ks,L,n); %(t,alpha, mu,v,d,x,boundary,pulse_time, n)
ft = sr_FADE(t,Ks,L,n); %(t,alpha, mu,v,d,x,boundary,pulse_time, n)
%% visualzation
% figure(1)
loglog(t,ft,'k','LineWidth',1)
% semilogy(t,ft,'r','LineWidth',1)
% plot(t,ft,'b','LineWidth',1)
hold on
plot(t,y,'ko','LineWidth',1)
% plot(t,y,'r','LineWidth',1.5)
% legend('Synthetic data','Recovered model')
xlabel('Time')
ylabel('Normalized concentration')
axis([-inf,inf,0.001,inf])