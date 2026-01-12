cleanData = rmmissing(MADE2);
t = cleanData(:,1);
y = cleanData(:,2);
t = table2array(t);
y = table2array(y);
loglog(t,y,'o')
%%
filename = 'river014.mat';
load(filename)
pp = csaps(t, y, 0.5); % 平滑参数（0~1，越大越贴近原始数据）
t_interp = t;
y_spline = fnval(pp, t_interp);
plot(t,y,'o'); hold on
plot(t_interp,y_spline)
t = t_interp; 
y = y_spline;
filename1 = ['smooth',filename];
save(filename1,'t','y')
axis([40,100,0,inf])