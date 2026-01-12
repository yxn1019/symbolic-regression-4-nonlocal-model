clear
filename = 'river051.mat';
load(filename)
% y = y/sum(y);
letters = regexp(filename, '[A-Za-z]+', 'match');
letters = letters{1};
num = regexp(filename, '\d+', 'match');
num = str2double(num{1}); 
if num==2; num=1; end
x = logspace(2,3,1000);
% x = t;

if strcmp(letters, 'tr')
    x0 = x/num;
%     y0 = 0.11053087 * x0.^(7.0126648./x0 - 1); 
    y0 = exp(-0.029645307./x0)*6.4882726e-5./x0.^2.1370769;
    y0 = y0/num;
    y0 = y0*300; 
    y = y*300;
elseif strcmp(letters, 'br')
    x0 = x/num;
    y0 =  7.6203547./((x0.^2.5310283).*exp(0.0292647./x0));
    y0 = y0/num;
%     y0 = 1./((21.516378)*x0.^2.4714448.*((exp(0.6866878./x0).*exp(x0*0.016769402)))); 
elseif strcmp(letters, 'saa')
    x0 = num./x;
    y0 = (x0 .^ 0.74582) ./ exp(x0); 
    y0 = num*y0./x.^2;
elseif strcmp(letters, 'sediment')
    x0 = x/num;
%     y0 = x0.^3.8919938*11.846161./exp(x0);  
%     y0 = (9.066056 ./ (x0 .^ -4.067767)) ./ exp(x0); % best
    y0 = (x0 - 3.5054746 + 13.426684./x0).^(-2.4587057); y0 = y0/num * 1.2081e+03; % sum(p)=1.2081e+03,sediment01
%     y0 = (x0 - 3.0004573 + 12.694672./x0).^(-2.7160604); y0 = y0/num * 2015.6316148; %sediment1
%     filename = 'sediment01';load(filename); m = sum(y);
else
%     error('please input correct filename')
%     if num ==51; num = 5.1; end
    x0 = num./x;
    y0 = x0.^(x0*(-190.09575) - 13.509841)*3.380207e-39;
    y0 = num*y0./x.^2;
end

% 画图
figure;
% semilogy(t,y,'o')
loglog(t,y/max(y),'o')
% plot(t,y,'ro')
hold on
plot(x,y0/max(y0)); 
if strcmp(letters, 'saa') 
    xlabel('Time');
%     ylabel('Normalized concentration');
    ylabel('Concentration');
else
    xlabel('Location (m)');
    ylabel('Normalized Mass');
%     ylabel('Concentration (mg/L)');
end
lgd = legend('Experimental concentration', 'Recovered model');
set(lgd, 'FontName', 'Times New Roman','FontSize', 10);
annotation('textbox', [0.5 0.5 0.2 0.1], ...
           'String', ['T=',num2str(num),'d'], ...
           'FontName', 'Times New Roman', ...
           'FontSize', 12, ...
           'EdgeColor', 'none');
set(gca, 'FontName', 'Times New Roman');% 统一设置坐标轴标签字体
% axis([1e-2,900,1e-2,300]) % br
% axis([1e-2,1000,1e-5,0.2]) % tr
% axis([-inf,inf,1e-4,2.5])
