kill
wd = 'C:\Users\KLN\Documents\projects\nichols';
cd(wd)
load('crispData.mat')
humtopTopcics = [29 97 76 21 23 66 72 34 78 10];
% genre codes follows numeric order of unique(G)
genreCodes = {'Confucianism' 'Mozi' 'Daoism' 'Legalism' 'School of Names' ...
    'Military' 'Mathematics' 'Miscellaneous' 'History' 'Classics' ...
    'Etymology' 'Medical' 'Excavated'};
% variables
    % tWeight2: topic weights with docs in rows and topics in columns (order on tNum2)
    % tNum2: topic numbers
    % comment: mallet normalizes weights 
        % iow: tWeight2 = (W - min2(W))/(max2(W) - min2(W)), where W is the original weight matrix  
%%% working variables     
feat = tWeight2;
% number of topics
m = size(feat,2);
% number of documents
n = size(feat,1);
%% unsupervised


% k-means
k = length(unique(date));
[idx, c, sumd, d] = kmeans(feat,k,'Distance','sqeuclidean','Replicates',10);


%% supervised
dateLabel = strread(num2str(date'),'%s'); %#ok<DSTRRD>
per = unique(dateLabel);
c = size(per,1);
% full set
inputs = feat';
targets = zeros(n,c);
size(targets)
for i = 1:length(per)   
    targets(:,i) = double(strcmp(per{i},dateLabel));    
end
% inspect features and classes
figure(2)
targets = targets';
subplot(221),
imagesc(targets); colormap('bone'); colormap(flipud(colormap));
xlabel('documents')
subplot(222),
imagesc(inputs);xlabel('documents');ylabel('features');
hold on
subplot(212), hist(date,c); xlabel('year')
% create network
hiddenLayerSize = 8;
net = patternnet(hiddenLayerSize);
% set up division of data
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;
% suppress gui
% net.trainParam.showWindow = false;
% train network
rng(0) 
[net,tr] = train(net,inputs,targets);
% performance on training set
outputs = net(inputs);% compute network output
errors = gsubtract(targets,outputs);% compute error
figure, plotperform(tr)
% plot confusion matrix
plotconfusion(targets,outputs)
