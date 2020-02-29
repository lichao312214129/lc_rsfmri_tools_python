% function machineLearningForLowDimensionData()
% 用来对降维后的数据进行机器学习
% 此代码用于多分类
%% =================================================================
% input
k=10;
trainingDataPath_original='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\trainingData_originalALFF.xlsx';
trainingDataPath='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\trainingData.xlsx';
testDataPath='D:\WorkStation_2018\WorkStation_2018_11_machineLearning_Psychosi_ALFF\Data\testData.xlsx';

if_cv=1;
ifSaveModel=0;
ifShow=0;
ifOriginalALFF=0;
ifTrainingFinalModel=1;
if_save_predict_label=1;
if_save_figure=0;
%% =================================================================
% load data
if ifOriginalALFF
    [training_data,~]=xlsread(trainingDataPath_original);
    [~,training_label]=xlsread(trainingDataPath);
else
    [training_data,training_label]=xlsread(trainingDataPath);
    training_data=training_data(:,end-9:end);%选择n维的数据
end
[test_data,testLabel_struct]=xlsread(testDataPath);
test_data=test_data(1:282,end-9:end);%只选择病人组的n维的数据

% extract label
training_label=training_label(2:end,2);
training_label_num=zeros(numel(training_label),1);
unique_training_label=unique(training_label);
n_label=numel(unique_training_label);
for i=1:n_label
    logic_loc=cellfun(@(x) ismember(x,unique_training_label{i}),training_label);
    training_label_num(logic_loc)=i;
end

testLabel_struct=testLabel_struct(2:end,1);
testLabel_struct= cellfun(@(x) strsplit(x,'-'),testLabel_struct,'UniformOutput',false);
testLabel=cellfun(@(x) str2double(x(1)), testLabel_struct);
testFolder=cellfun(@(x) x(2), testLabel_struct);

% step2： 标准化效果最好
% mean(trainingData(:));std(trainingData(:))
% mean(testData(:));std(testData(:))
% [trainingData,testData]=lc_standardization(trainingData,testData,'scale');

training_data=zscore(training_data,0,1);
test_data=zscore(test_data,0,1);

% training_data=mapminmax(training_data);
% test_data=mapminmax(test_data);

% [training_data] = lc_demean(training_data,2);
% [test_data] = lc_demean(test_data,2);
%% =================================================================
% 交叉验证
Model=cell(5,1);
Decision=cell(k,1);
AUC=zeros(k,1);
Accuracy=zeros(k,1);
Sensitivity=zeros(k,1);
Specificity=zeros(k,1);
PPV=zeros(k,1);
NPV=zeros(k,1);


ind_kfolder=crossvalind('Kfold', length(training_data), k);

t = templateNaiveBayes();
% t=templateTree('Surrogate','on');
% t=templateLinear();
% t=templateKNN();
% t=templateSVM('KernelFunction','polynomial');
if if_cv
%     pool = parpool; % Invoke workers
%     options = statset('UseParallel',true);
target=[];
output=[];


    for i=1:k
        fprintf('%d/%d\n',i,k);
        % step1：将数据分成训练样本和测试样本（分别从患者和对照组中抽取，目的是为了数据平衡）
    %     n_patients=sum(training_label==1);
    %     n_controls=sum(training_label~=1);
    %     
    %     indices_p = crossvalind('Kfold', n_patients, k);
    %     indices_c = crossvalind('Kfold', n_controls, k);
    %     indiceCell={indices_c,indices_p};
    %     
    %     [train_data,test_data,Train_label,Test_label]=...
    %         BalancedSplitDataAndLabel(trainingData,training_label,indiceCell,i);

        % step2： 标准化或者归一化
        %     [train_data,test_data]=lc_standardization(train_data,test_data,'scale');         
        model= fitcecoc(training_data(ind_kfolder~=i),training_label_num(ind_kfolder~=i),...
                        'Learners',t,'Coding','onevsone');

        % estimate mode/SVM
        [predict_label, dec_values] = predict(model,training_data(ind_kfolder==i));
        
        %save label
        target=[target;predict_label];
        output=[output;training_label_num(ind_kfolder==i)];
%         Model{i}=model;
%         Decision{i}=dec_values(:,1);
         Decision{i}=dec_values;
%         [accuracy,sensitivity,specificity,ppv,npv]=Calculate_Performances(predict_label,training_label_num(ind_kfolder==i));
%         Accuracy(i) =accuracy;
%         Sensitivity(i) =sensitivity;
%         Specificity(i) =specificity;
%         PPV(i)=ppv;
%         NPV(i)=npv;
%         [AUC(i)]=AUC_LC(training_label_num(ind_kfolder==i),dec_values(:,2));
    end
end

% plot confusion matrix
figure;
target_m=zeros(5,numel(target));
output_m=zeros(5,numel(target));
for i =1:numel(target)
    target_m(target(i),i)=1;
    output_m(output(i),i)=1;
end
plotconfusion(target_m,output_m);

fprintf('=======CV Done!======\n')
%% =================================================================
% save results
if ifSaveModel
    save('Model.mat','Model');
end
%% =================================================================
% show results
if ifShow
    performance={AUC,Accuracy,Sensitivity,Specificity,PPV,NPV};
    mytitle={'AUC','Accuracy','Sensitivity','Specificity','PPV','NPV'};
    figure
    for i=1:6
        subplot(2,3,i)
        plot(performance{i},'-o');
        title(mytitle{i});
    end
    % save figure
    if if_save_figure
        print(gcf,'-dtiff','-r300',['CV'])
    end
    
    % mean
    figure
    Balance_Accuracy=(Sensitivity+Specificity)./2;
    bar_errorbar3( {AUC,Accuracy,Balance_Accuracy,Sensitivity,Specificity,PPV,NPV} )
    title('mean performances');
    % save figure
    if if_save_figure
        set (gcf,'Position',[100,50,600,500], 'color','w')
        print(gcf,'-dtiff','-r300',['mean performances'])
    end

end
%% =================================================================
% 如果在训练集内交叉验证效果不错，那么就用总训练集建模，然后在测试集上测试
if ifTrainingFinalModel
    try
        model_allTrainingData= fitcecoc(training_data,training_label_num,...
            'Learners',t,'Coding','onevsall');
    catch
        %
        model_allTrainingData = fitensemble(training_data,training_label_num,'AdaBoostM2',100,t);
        L = resubLoss(model_allTrainingData);
        
    end
     
    [predictLabel_testData, dec_values] = predict(model_allTrainingData,test_data);
    

%     save predictLabel_testData
    if if_save_predict_label
        xlswrite('predict_label_multilabel.xlsx',predictLabel_testData)
    end
%% 打印构成   
    % 训练样本
    fprintf('训练集为a的数:%d，比例为：%.2f\n',sum(training_label_num==1),sum(training_label_num==1)/581);
    fprintf('训练集为b的数:%d，比例为：%.2f\n',sum(training_label_num==2),sum(training_label_num==2)/581);
    fprintf('训练集为c的数:%d，比例为：%.2f\n',sum(training_label_num==3),sum(training_label_num==3)/581);
    fprintf('训练集为d的数:%d，比例为：%.2f\n',sum(training_label_num==4),sum(training_label_num==4)/581);
    fprintf('训练集为e的数:%d，比例为：%.2f\n\n',sum(training_label_num==5),sum(training_label_num==5)/581);
    
    % 测试样本
    fprintf('预测为a的数:%d，比例为：%0.2f\n',sum(predictLabel_testData==1),sum(predictLabel_testData==1)/282);
    fprintf('预测为b的数:%d，比例为：%.2f\n',sum(predictLabel_testData==2),sum(predictLabel_testData==2)/282);
    fprintf('预测为c的数:%d，比例为：%.2f\n',sum(predictLabel_testData==3),sum(predictLabel_testData==3)/282);
    fprintf('预测为d的数:%d，比例为：%.2f\n',sum(predictLabel_testData==4),sum(predictLabel_testData==4)/282);
    fprintf('预测为e的数:%d，比例为：%.2f\n',sum(predictLabel_testData==5),sum(predictLabel_testData==5)/282);
end

function  bar_errorbar3( Matrix )
%% =================================================================
showXTickLabels=1;
showYLabel=0;
showLegend=0;
%% =================================================================
Mean=cell2mat(cellfun(@(x) mean(x,1),Matrix,'UniformOutput',false)')';
Std=cell2mat(cellfun(@(x) std(x),Matrix,'UniformOutput',false)')';

h = bar(Mean,0.6,'EdgeColor','k','LineWidth',1.5);
f = @(a)bsxfun(@plus,cat(1,a{:,1}),cat(1,a{:,2})).';%获取每一个柱状图中线的x坐标
coordinate_x=f(get(h,{'xoffset','xdata'}));
hold on
%画误差线
for i=1:numel(Mean)
    if Mean(i)>=0
        line([coordinate_x(i),coordinate_x(i)],[Mean(i),Mean(i)+Std(i)],'linewidth',2);
    else
        line([coordinate_x(i),coordinate_x(i)],[Mean(i),Mean(i)-Std(i)],'linewidth',2);
    end
end
grid on
box off
ax=gca;
% x轴的label
ax.XTickLabels = ...
    {'AUC','Accuracy','Balance-Accuracy','Sensitivity','Specificity','PPV','NPV'};
set(ax,'Fontsize',15);
ax.XTickLabelRotation = 45;

% y轴的labe
ylabel('dALFF','FontName','Times New Roman','FontWeight','bold','FontSize',20);
% legend
% h=legend('HC','SZ','BD','MDD','Orientation','horizontal','Location','best');%根据需要修改
% set(h,'Fontsize',15);%设置legend字体大小
% set(h,'Box','off');
% h.Location='best';
box off
end
