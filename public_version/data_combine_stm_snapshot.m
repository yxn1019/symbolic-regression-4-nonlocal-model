clear
% data = [];
addpath('../dataset')
filen = ["tr27.mat","tr132.mat","tr224.mat","tr328.mat"]; % The filen depends on the name of data
data = [];
for i = 1:length(filen) % this loop combine multiple measurements and convert them to pdf of velocity
    filename = filen(i);    % For the first run, choose between tr132 and br202
    load(filename)
    % y = y/sum(y);
    num = regexp(filename, '\d+', 'match');
    num = str2double(num{1}); % Time (days)
    x = t;
    t0 = num;
    v = x ./ t0; 
    % Smoothing using spline interpolation
    pp = csaps(x, y, 1); % Smoothing parameter (0~1, higher value follows original data more closely)
    x_interp = t0 * v;
    y_spline = t0 * fnval(pp, x_interp);
    data = [data; [v, y_spline]];
    data = data(data(:, 1) >= 0, :); % Remove elements with negative position
    data = data(data(:, 2) >= 0, :); % Remove elements with negative concentration
    data = sortrows(data, 1);
end

% save([filename, '.mat'], 't', 'y', '-append')
% t = v; y = y_spline; save(['saa', filename, '.mat'], 't', 'y')
%% Plotting
% figure;
loglog(v, y, 'bo', 'DisplayName', 'Noisy data');
hold on;
semilogy(v, y_spline / t0, '-', 'LineWidth', 1.5, 'DisplayName', 'pdf');  % The stored data and the plotted graph are different; this is for comparison with the source data
hold on
legend;
title('Spline Interpolation Smoothing');