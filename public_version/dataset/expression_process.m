% 输入数据
clear
load('br370')
data = [t,y];

% 提取x和浓度c
x = data(:, 1);  % 第一列为x坐标
c = data(:, 2);  % 第二列为浓度值

% 计算质心位置
numerator = sum(x .* c);    % 分子 Σ(x_i * c_i)
denominator = sum(c);      % 分母 Σc_i
centroid_x = numerator / denominator;

% 输出结果
disp(['质心位置: ', num2str(centroid_x)]);
% disp(['质心 vt = ', num2str(x_centroid_t1)]);
%%
t = [202; 279; 370];
x_centroid = [10.0755; 11.2843; 12.809];
p = polyfit(t, x_centroid, 1);  % 一阶线性拟合
v = p(1);  % 斜率即平均速度
disp(['平均速度 v = ', num2str(v), ' 单位/时间']);