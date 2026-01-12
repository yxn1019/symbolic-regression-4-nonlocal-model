% clear
% data = [];
filename = 'tr132';    %第一次运行要从tr132和br202中选择
load(filename)
% y = y/sum(y);
num = regexp(filename, '\d+', 'match');
num = str2double(num{1}); % 时间/d
x = t;
t0 = num;
v = x./t0; 
% 样条插值平滑
pp = csaps(x, y, 1); % 平滑参数（0~1，越大越贴近原始数据）
x_interp = t0 * v;
y_spline = t0 * fnval(pp, x_interp);
if t0 == 22 || t0 == 1 || t0==14
    data = []; 
end
data = [data;[v,y_spline]];
data = data(data(:, 1) >= 0, :); %剔除位置为负的元素
data = data(data(:, 2) >= 0, :); %剔除浓度为负的元素
data = sortrows(data, 1);
% if num == 328/24 || num == 370/24; t = data(:,1); y = data(:,2); end

% save([filename,'.mat'], 't','y', '-append')
% t=v;y=y_spline;save(['saa',filename,'.mat'], 't','y')
%% 绘图
% figure;
loglog(v, y, 'bo', 'DisplayName', '带噪数据');
hold on;
semilogy(v, y_spline/t0, '-', 'LineWidth', 1.5, 'DisplayName', filename);  %实际存储的和画出来的图不一样，这里是为了和源数据进行对比
hold on
legend;
title('样条插值平滑');