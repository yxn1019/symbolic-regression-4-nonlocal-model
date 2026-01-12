function ft = sr_FADE(t,Ks,L,n)
%     if nargin==6 
%         boundary=1;n=1;fprintf('warn: without influx can only use 1st boundary.\n')
%     elseif nargin==7 && boundary~=1
%         fprintf('error: please input inflow flux!\n')
%     end
%%%%%%%%%%%._._._._. 构造Laplace空间的解析解：fs    =.==.==.==.==.==.=
    syms s
%     g_s = ((mu+s)^alpha-mu^alpha);
%     if L == 1.4
%         load('river014.mat')
%     %     Ks =  - log(2.1961076 * ((s + 0.28531173).^(-4.836802) - 0.023395305)) ./ L;Ks = Ks + 42*s/L ;
%         Ks = (s)^0.93403035*36.05492 - 5.0189967; % 不需要乘exp()
%     %     Ks = ((s ^ 0.65155345) * 6.446834) + -5.253608;   Ks = Ks + 42*s/L ;% 需要乘exp()%   
%     %     + log(sum(y))/L
%     elseif L == 3.1
%         load('river031.mat')
%     %     Ks = (s + 0.0026856377)*29.06269 - 0.58100677/(s + 0.25228798) ;  
%         Ks = -0.5358925/(s + 0.24222377); 
%         Ks = Ks + 90*s/L ; % no normalization
%     elseif L == 5.1
%         load('river051.mat')
%         Ks = -1.341868*exp(-4.914254*s^0.92856205); % 5.1
%     %     Ks = 0.5925046 - 0.44701213/(s + 0.23259251); % no normalization
%         Ks = Ks + 130*s/L ;
%     else
%         warning('Detection position is wrong!')
%     end
    fs = n*exp(-L*Ks);
%     v=0.029; d=0.000467;
%     fs = 27.85*exp((5.1*(v - (v^2 + 4*d*s^0.978)^(1/2)))/(2*d));
%%%%%%% Laplace逆变换
    % ft = talbot_inversion(fs,t);
    ft = nil(fs,t);
end
