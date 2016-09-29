% load vector space model
clear
cd C:\Users\KLN\Documents\projects\slingerland\pilot
load wordDocMat.mat
nw = length(inventory);% number of characters
inventoryNumCell = num2cell(1:nw);% numeric indices for inventory
    % e.g., inventory{cell2mat(inventoryNumCell(3))} % get 3rd character
    % from inventory
% check that no stopwords are present in inventory
load stopWords.mat    
for i = 1:length(stopWords)    
    l = sum(strcmp(stopWords(i), inventory));
    if l ~= 0
        disp(stopWords(i))
        disp(strmatch(stopWords(i), inventory, 'exact')) %#ok<MATCH3>
    end
end
% clean
clear stopWords i l
%% write stopword list
t = table(stopWords,'VariableNames',{'Stopwords'});
writetable(t,'stopwords.csv')


fileID = fopen('stopWords.dat','w');
formatSpec = '%s\n';
[nrows,ncols] = size(stopWords);
for row = 1:nrows
    fprintf(fileID,formatSpec,stopWords{row,:});
end

%%
wd = 'C:\Users\KLN\Documents\projects\slingerland\associations';
cd(wd);
load('associationterms.mat')
%%
idx1 = zeros(size(query1));
for i = 1:length(query1)
    strmatch(query1(i),inventory,'exact')
end
%% problem with
strmatch(query1(2),inventory,'exact')
% in stopwords
strmatch(query1(2),stopWords,'exact')% yes
%%
%% alternate data set
wd = 'C:\Users\KLN\Documents\projects\humclass';
cd(wd)
load('chinese2Test.mat')
wd = 'C:\Users\KLN\Documents\projects\slingerland\associations';
cd(wd);
load('associationterms.mat')
inventory = vocabfull;
% query term 1
idx1 = zeros(size(query1))';
for i = 1:length(query1)
    idx1(i) = strmatch(query1(i),inventory,'exact');
end
% query term 2 (body terms)
idx2 = zeros(size(query2))';
for i = 1:length(query2)
    idx2(i) = strmatch(query2(i),inventory,'exact');
end
% correlate vector space
q1q2cor = zeros(length(idx1),length(idx2),2);
for j = 1:length(idx1)
    x = tdm(:,idx1(j));
    for jj = 1:length(idx2)
        y = tdm(:,idx2(jj));
        [r,p] = corrcoef(x,y);
        q1q2cor(j,jj,1) = r(2,1);
        q1q2cor(j,jj,2) = p(2,1);
    end
end
disp(q1q2cor)% strongly correlated, but might just reflect varying length
% plot document length & central tendency
hf = figure(1); h1 = bar(sum(tdm,2),'FaceColor',[0 0 0], ...
    'EdgeColor',[1 1 1]); hline(median(sum(tdm,2)),'r-')
% compare word frequency and length
doclen = sum(tdm,2);
hf3 = figure(3);
for i = 1:length(query1)
    subplot(3,4,i), % h3 = scatter(tdm(:,idx1(i)),doclen,'.k');
    bagplot([tdm(:,idx1(i)) doclen])
    [r,p] = corrcoef(tdm(:,idx1(i)),doclen);
    title(['r = ' num2str(r(2,1)) ', p = ' num2str(p(2,1))] )
    xlabel(query1(i)); ylabel('Document length')
    %l = lsline;
end
%% 1K slice corpus for length standardization
% return to humclass library
wd = 'C:\Users\KLN\Documents\projects\humclass';
cd(wd)
load('chinese2Test.mat')
% slice in 1000 chars without moving window
[Y, docslen] = sliceTok(docstokens,1000);
% rearrange with class var
[Ycell, YcellFile] = sliceCell(Y,filename);
% build new tdm on slices
[to_1K, vocabfull_1K] = dtmTok(Ycell);
file_1K = YcellFile;
%save('chinese2Test_1K.mat','to_1K','vocabfull_1K','file_1K')
%% 1C slice corpus for length standardization
wd = 'C:\Users\KLN\Documents\projects\humclass';
cd(wd)
load('chinese2Test.mat')
% slice in 100 chars without moving window
[Y, docslen] = sliceTok(docstokens,100);
% rearrange with class var
[Ycell, YcellFile] = sliceCell(Y,filename);
% build new tdm on slices
[to_1C, vocabfull_1C] = dtmTok(Ycell);
file_1C = YcellFile;
save('chinese2Test_1C.mat','to_1C','vocabfull_1C','file_1C')
%% rerun on 1K corpus
wd = 'C:\Users\KLN\Documents\projects\slingerland\associations';
cd(wd);
load('chinese2Test_1K.mat')
load('associationterms.mat')
inventory = vocabfull_1K;
% query term 1
idx1 = zeros(size(query1))';
for i = 1:length(query1)
    idx1(i) = strmatch(query1(i),inventory,'exact');
end
% query term 2 (body terms)
idx2 = zeros(size(query2))';
for i = 1:length(query2)
    idx2(i) = strmatch(query2(i),inventory,'exact');
end
% correlate sliced vector space
q1q2cor_1K = zeros(length(idx1),length(idx2),2);
for j = 1:length(idx1)
    x = to_1K(:,idx1(j));
    for jj = 1:length(idx2)
        y = to_1K(:,idx2(jj));
        [r,p] = corrcoef(x,y);
        q1q2cor_1K(j,jj,1) = r(2,1);
        q1q2cor_1K(j,jj,2) = p(2,1);
    end
end
disp(q1q2cor_1K)% strongly correlated, but might just reflect varying length
for i = 1:size(q1q2cor_1K,2)
    q1q2cor_1K(:,i,3) = zeroOneScale(q1q2cor_1K(:,i,1));
end
hf4 = figure(4); h4 = bar(q1q2cor_1K(:,:,3),'stacked');
set(gca,'xticklabel',query1)
hl = legend(query2,'location','bestoutside'); legend boxoff
xlim([0.1 12.9])
%ylim([0 1.1]);
box off
set(gcf,'Paperpositionmode','auto','Color',[1 1 1]);
a = findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines = findall(a,'Type','line');
alltext = findall(a,'Type','text');
set(allaxes,'FontWeight','Bold','LineWidth',1.5,...
'FontSize',12);
colormap(gray)
%% run associations between control terms to test different vector space models
% 1K mdl
%wd = 'C:\Users\KLN\Documents\projects\slingerland\associations';
wd = '/home/kln/projects/slingerland/associations';
cd(wd)
load('chinese2Test_1K.mat')
load('associationterms.mat')
% load full model
%cd('C:\Users\KLN\Documents\projects\humclass');
cd('/home/kln/projects/humclass');
load('chinese2Test.mat')
cd(wd)
% calculate and compare association measures for 1K and full
ctrcorr_full = zeros(size(control));
ctrcorr_1K = zeros(size(control));
dist_1K = zeros(length(control),3);
for i = 1:size(control,1)
    xidx = strmatch(control(i,1), vocabfull_1K,'exact');
    yidx = strmatch(control(i,2), vocabfull_1K,'exact');
    [r, p] = corrcoef([to_1K(:,xidx) to_1K(:,yidx)]);
    ctrcorr_1K(i,1) = r(2,1);
    ctrcorr_1K(i,2) = p(2,1);
    [r, p] = corrcoef([tdm(:,xidx) tdm(:,yidx)]);
    ctrcorr_full(i,1) = r(2,1);
    ctrcorr_full(i,2) = p(2,1);
    % euclidean distance between vectors in 1K mdl
    v = to_1K(:,xidx)' - to_1K(:,yidx)';
    d = sqrt(v * v');
    dist_1K(i,1) = d;
    % jaccard index;
    jd = pdist([to_1K(:,xidx)';to_1K(:,yidx)'],'jaccard');  % jaccard distance
    ji = 1 - jd; 
    dist_1K(i,2) = ji; 
    % cosine
    cosd = pdist([to_1K(:,xidx)';to_1K(:,yidx)'],'cosine'); 
    dist_1K(i,3) = cosd;
end
disp(dist_1K)
%disp([ctrcorr_1K(:,1) ctrcorr_full(:,1)])
% full model does not catch word contrast, everything is just associated
% due to document size, but 1K finds contrast associations
%% clustering on the control terms from 1K model
% build vector space of control terms
[ctrlist,ix] = unique(reshape(control,1,length(control)*2));
ctrlisttr = reshape(controltr,1,length(controltr)*2);
    ctrlisttr = ctrlisttr(ix); ctrlisttr = regexprep(ctrlisttr,'_','');
idx = zeros(size(ctrlist));
for i = 1:length(idx);
    idx(i) = strmatch(ctrlist(i),vocabfull_1K,'exact');
end
ctrmat = to_1K(:,idx);
ctrd = squareform(pdist(ctrmat','cosine'));
ctrlink = linkage(ctrd,'average');
ctrlisttr = strcat(ctrlist,':',ctrlisttr);
% Ted's correction google translate
    %ctrlisttr{3} = '?:human/person';
    %ctrlisttr{16} = '?:star';
    %ctrlisttr{18} = '?:moon/month'; 
    %ctrlisttr{26} = '?:see';
    %ctrlisttr{27} = '?:run';
    %ctrlisttr{31} = '?:fly';

hf = figure(1);
%subplot(121),
df = dendrogram(ctrlink,length(ctrlist),'Labels',ctrlisttr, ...
    'orientation', 'left','ColorThreshold','default');
set(df,'LineWidth',2)
title('Control terms')
% clustering on the query terms from 1K model
querylist = [query1' query2'];
querylisttr = [query1tr' query2tr'];
idx = zeros(size(querylist));
for i = 1:length(idx);
    idx(i) = strmatch(querylist(i),vocabfull_1K,'exact');
end
qrymat = to_1K(:,idx);
qryd = squareform(pdist(qrymat','cosine'));
qrylink = linkage(qryd,'average');
qrylisttr = strcat(querylist,':',querylisttr);
set(gcf,'Paperpositionmode','auto','Color',[1 1 1]);
a = findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines = findall(a,'Type','line');
alltext = findall(a,'Type','text');
set(allaxes,'FontWeight','Bold','LineWidth',1.5,...
'FontSize',12)
figSize(hf,15,15)
%saveFig(strcat(wd,'/dendro_assoc_part1.png'))
%saveFig(strcat(wd,'/dendro_assoc_part1_bw.png'))
hf2 = figure(2);
%subplot(122),
df2 = dendrogram(qrylink,length(querylist),'Labels',qrylisttr, ...
    'orientation', 'left','ColorThreshold','default');
title('Query terms')
set(df2,'LineWidth',2)
set(gcf,'Paperpositionmode','auto','Color',[1 1 1]);
a = findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines = findall(a,'Type','line');
alltext = findall(a,'Type','text');
set(allaxes,'FontWeight','Bold','LineWidth',1.5,...
'FontSize',12)
figSize(hf2,10,15)
%saveFig(strcat(wd,'/dendro_assoc.png'))
%saveFig(strcat(wd,'/dendro_assoc_part2.png'))
%saveFig(strcat(wd,'/dendro_assoc_part2_bw.png'))
%%









%% compare full corpus to 1K sliced corpus
disp(q1q2cor(:,:,1))
disp(q1q2cor_1K(:,:,1))
% standardize coefficients to compare columns
for i = 1:size(q1q2cor,2)
    q1q2cor(:,i,3) = zeroOneScale(q1q2cor(:,i,1));
    q1q2cor_1K(:,i,3) = zeroOneScale(q1q2cor_1K(:,i,1));
end
disp(q1q2cor(:,:,3))
disp(q1q2cor_1K(:,:,3))
% plot association scores for full text and 1K slices
for j = 1:3
    hf2 = figure(2); subplot(1,3,j),f = bar([q1q2cor(:,j,3),q1q2cor_1K(:,j,3)]);
    title(query2(j))
    set(gca,'xticklabel',query1)
    ylabel('Standardized Pearson')
    colormap(gray); ylim([0 1.1])
    [~,p] = ttest2(q1q2cor(:,j,3),q1q2cor_1K(:,j,3)); disp(p)
end
%% 1C vector space
wd = 'C:\Users\KLN\Documents\projects\slingerland\associations';
cd(wd)
load('associationterms.mat')
wd = 'C:\Users\KLN\Documents\projects\humclass';
cd(wd)
load('chinese2Test_1C.mat')

% build vector space of control terms
[ctrlist,ix] = unique(reshape(control,1,length(control)*2));
ctrlisttr = reshape(controltr,1,length(controltr)*2);
    ctrlisttr = ctrlisttr(ix); ctrlisttr = regexprep(ctrlisttr,'_','');
idx = zeros(size(ctrlist));
for i = 1:length(idx);
    idx(i) = strmatch(ctrlist(i),vocabfull_1C,'exact');
end
ctrmat = to_1C(:,idx);
ctrd = squareform(pdist(ctrmat','cosine'));
ctrlink = linkage(ctrd,'average');
ctrlisttr = strcat(ctrlist,':',ctrlisttr);
hf = figure(1);
%subplot(121),
df = dendrogram(ctrlink,length(ctrlist),'Labels',ctrlisttr, ...
    'orientation', 'left','ColorThreshold','default');
set(df,'LineWidth',2)
title('Control terms')
% query terms
querylist = [query1' query2'];
querylisttr = [query1tr' query2tr'];
idx = zeros(size(querylist));
for i = 1:length(idx);
    idx(i) = strmatch(querylist(i),vocabfull_1C,'exact');
end
qrymat = to_1C(:,idx);
qryd = squareform(pdist(qrymat','cosine'));
qrylink = linkage(qryd,'average');
qrylisttr = strcat(querylist,':',querylisttr);
set(gcf,'Paperpositionmode','auto','Color',[1 1 1]);
a = findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines = findall(a,'Type','line');
alltext = findall(a,'Type','text');
set(allaxes,'FontWeight','Bold','LineWidth',1.5,...
'FontSize',12)
figSize(hf,15,15)
%% saveFig(strcat(wd,'/dendro_assoc_part1.png'))
hf2 = figure(2);
%subplot(122),

df2 = dendrogram(qrylink,length(querylist),'Labels',qrylisttr, ...
    'orientation', 'left')%,'ColorThreshold',0.568 * max(qrylink(:,3)));
title('Query terms')
set(df2,'LineWidth',2)
set(gcf,'Paperpositionmode','auto','Color',[1 1 1]);
a = findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines = findall(a,'Type','line');
alltext = findall(a,'Type','text');
set(allaxes,'FontWeight','Bold','LineWidth',1.5,...
'FontSize',12)
figSize(hf2,10,15)
