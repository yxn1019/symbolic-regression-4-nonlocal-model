% data_file = ''
x = data328.t;
y = data328.y;
pp = csaps(x, y, 1); % 平滑参数（0~1，越大越贴近原始数据）
x_interp_328 = linspace(0.01,27,100);
y_interp_328 = fnval(pp, x_interp_328);
loglog(x,y,'o',x_interp_328,y_interp_328,'x')
%%
plot3(x,t,y,'o')


%% 
load('river014')
pp = csaps(t, y, 0.3); % 平滑参数（0~1，越大越贴近原始数据）
t_interp = linspace(0.01,100,100);
y_interp = fnval(pp, t_interp);
y_interp(1:41) = 0;
plot(t,y, 'o',t_interp, y_interp)

