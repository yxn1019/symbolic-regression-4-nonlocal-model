clear
% data = [];
addpath('../dataset')
filen = ["saa10.mat","saa30.mat","saa50.mat"]; % The filen depends on the name of data
data = [];
for i = 1: length(filen) % this loop combine multiple measurements and convert them to pdf of velocity
    filename = filen(i);
    load(filename)
    num_str = regexp(filename, '\d+', 'match');
    num = str2double(num_str{1}); % Time in days
    t = t(y > 1e-4);
    y = y(y > 1e-4);
    if num == 14
        num = 1.4;
    end
    x0 = num;
    v = x0 ./ t; 

    % Smoothing using spline interpolation
    pp = csaps(t, y, 0.9); % Smoothing parameter (0~1, higher value follows original data more closely)
    t_interp = x0 ./ v;
    y_spline = x0 ./ v.^2 .* fnval(pp, t_interp);
    data = [data; [v(:), y_spline(:)]];
    data = data(data(:, 1) > 0, :); % Remove elements with negative position
    data = data(data(:, 2) > 0, :); % Remove elements with negative concentration
    data = sortrows(data, 1);
% if num  328/24 || num  370/24; t = data(:,1); y = data(:,2); end
end
%% Plotting
% figure;
loglog(v, y, 'bo', 'DisplayName', 'Noisy data');
hold on;
semilogy(v, x0 * y_spline ./ t.^2, '-', 'LineWidth', 1.5, 'DisplayName', 'stm');
hold on
legend;
title('pdf');
% pp2 = csaps(data(:,1),data(:,2), 0.9); % Smoothing parameter (0~1, higher value follows original data more closely)
% t = data(:,1);
% y = fnval(pp2, t);
% y = data(:,2);
% save('saa1.mat', 't','y')
% t=v; y=y_spline; save(['saa',filename], 't','y')