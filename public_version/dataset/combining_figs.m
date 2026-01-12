% 需要显示的 .fig 文件名列表

figFiles = {'sediment_p_v.fig','sediment22.fig','sediment44.fig','sediment71.fig',...
            'sediment96.fig','sediment118.fig'};

nFig = numel(figFiles);          % 总共 6 张图
nRow = 3;                        % 想要的 subplot 行数
nCol = 2;                        % 想要的 subplot 列数

% 先创建一个干净的新 Figure 用来放 subplot
newFig = figure('Name','Combined Figures','Color','w');
clf(newFig);

for k = 1:nFig
    % ---------- 1. 打开老图 ----------
    oldFig = openfig(figFiles{k},'invisible');  % 不显示，只读
    oldAx   = findobj(oldFig,'Type','axes','-not','Tag','legend'); % 找到坐标轴
    % 如果图里有多条曲线、多个 patch 等，用 copyobj 一次性搬走
    % ---------- 2. 建新 subplot ----------
    newAx = subplot(nRow,nCol,k,'Parent',newFig);
    copyobj(get(oldAx,'Children'),newAx);       % 把老图里所有内容搬到新 Axes
    % 可选：复制坐标轴属性
    copyAxesProps(oldAx,newAx);
    close(oldFig);                              % 关掉老图
end

% ---------- 可选：把老图坐标轴属性也搬过来 ----------
function copyAxesProps(srcAx,dstAx)
    propList = {'FontName','FontSize','LineWidth',...
                'XLabel','YLabel','ZLabel',...
                'Title','XGrid','YGrid','ZGrid',...
                'XLim','YLim','ZLim',...
                'XScale','YScale','ZScale',...
                'XTick','YTick','ZTick',...
                'XTickLabel','YTickLabel','ZTickLabel',...
                'DataAspectRatio','PlotBoxAspectRatio'};
    for p = 1:numel(propList)
        try
            set(dstAx, propList{p}, get(srcAx,propList{p}));
        end
    end
end
