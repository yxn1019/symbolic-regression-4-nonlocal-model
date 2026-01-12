% clear
% data = [];
filename = 'river051.mat';
load(filename)
num = regexp(filename, '\d+', 'match');
num = str2double(num{1}); % 时间/d
t = t(y > 1e-4);
% t = t/60;
y = y(y > 1e-4);
x0 = num*100; %km
v = x0./t; %
% 样条插值平滑
pp = csaps(t, y, 0.9); % 平滑参数（0~1，越大越贴近原始数据）
t_interp = x0 ./ v;
y_spline = x0 ./v.^2 .* fnval(pp, t_interp);
if num == 14; data = []; end
data = [data;[v(:),y_spline(:)]];
data = data(data(:, 1) > 0, :); %剔除位置为负的元素
data = data(data(:, 2) > 0, :); %剔除浓度为负的元素
data = sortrows(data, 1);
% if num == 328/24 || num == 370/24; t = data(:,1); y = data(:,2); end


%% 绘图
% figure;
loglog(v, y, 'bo', 'DisplayName', '带噪数据');
hold on;
semilogy(v, x0*y_spline./t.^2, '-', 'LineWidth', 1.5, 'DisplayName', filename);
hold on
legend;
title('样条插值平滑');
pp2 = csaps(data(:,1),data(:,2), 0.9); % 平滑参数（0~1，越大越贴近原始数据）
t = data(:,1);
y = fnval(pp2, t);
% y = data(:,2);
% save('saa1.mat', 't','y')