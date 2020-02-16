clc;
clear;


A=xlsread('BB.xlsx'); %The data for traininig and testing

inputdata=A(1:750,1:6);  %Training input data
outputdata=A(1:150,7); %Training output data

inputdata1=inputdata';
outputdata1=outputdata';

%选连样本输入输出数据归一化[0,1]
[inputdata11,inputps]=mapminmax(inputdata1,0,1);

[outputdata11,outputps]=mapminmax(outputdata1,0,1);

P=inputdata11';
T=outputdata11';
%%  ex1 train a 100 hidden unit RBM and visualize its weights
%rand('state',0)
dbn.sizes = [30 30];    %RBM每个隐层有30个节点
opts.numepochs =  1;         %计算时根据输出误差返回调整神经元权值和阀值的次数
opts.batchsize = 80;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, P, opts);
dbn = dbntrain(dbn, P, opts);

% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights 

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);       %1个输出节点
nn.activation_function = 'sigm';  %激活函数sigmoid  sigm

%train nn
opts.numepochs = 40;
opts.batchsize = 50;   %1
nn = nntrain(nn, P(1:150,:), T, opts);

Ytrain=nnff1(nn,P(1:150,:));
Ytrain1=Ytrain';

%% BP网络训练
%初始化网络结构
inputn=Ytrain1;
outputn=T';
s1=40;%隐含层节点
disp('训练bp神经网络')
net=newff(inputn,outputn,s1);
net.trainParam.epochs=100;%训练次数
net.trainParam.lr=1;%学习率
net.trainParam.goal=0.03;%学习目标
net.trainParam.max_fail = 200;% 最小确认失败次数 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
%建立网络
%网络训练
net=train(net,inputn,outputn);
disp('结束训练bp神经网络')

%% ELMAN训练误差
% net=newelm(minmax(inputn),[40,1],{'tansig','tansig'});
% net.trainparam.show=100;%每迭代100次显示1次
% net.trainparam.epochs=3000;%最大迭代次数2000
% net.trainparam.goal=0.03;%迭代目标
% net=init(net);%初始化网络
% %网络训练
% [net,tr]=train(net,inputn,outputn);

%% RBF训练
% switch 2
% case 1 
%          
% % 神经元数是训练样本个数 
% spread = 1;                % 此值越大,覆盖的函数值就大(默认为1) 
% net = newrb(inputn,outputn);    
% % save BRPRBF net;
% case 2 
%      
% % 神经元数逐步增加,最多就是训练样本个数 
% goal = 0.013;                % 训练误差的平方和(默认为0) 
% spread =1;                % 此值越大,需要的神经元就越少(默认为1) 
% MN = size(P,2);% 最大神经元数(默认为训练样本个数) 
% DF = 1;                     % 显示间隔(默认为25) 
% net = newrb(inputn,outputn,goal,spread,MN,DF); 
%    
%     case 3     
% P = Ytrain1; 
% T = train_y1;  
% spread = 1;                % 此值越大,需要的神经元就越少(默认为1) 
% net = newgrnn(inputn,outputn,spread); 
%      
% end

%% 测试数据
input_test1=A(751:800,1:6);     %测试数据(仅测试一行)
output_test1=A(751:800,7);
input_test=input_test1';
output_test=output_test1';
test_x=mapminmax('apply',input_test,inputps,0,1)'; 
Test_x=nnff1(nn,test_x);

an=sim(net,Test_x');
%修改部分，输出预测值

BPoutput=mapminmax('reverse',an,outputps);

%% 输出对比
error2=BPoutput-output_test;
% MSE1=sum((BPoutput-output_test).^2)/length(BPoutput);
figure(1)
plot(BPoutput,'r-*')
hold on
%title('实际值与预测值拟合图','fontsize',10,'fontangle','normal')
plot(output_test,'b.-')
legend('预测值','实际值','Location','NorthEast');
xlabel('样本序列','fontsize',10)
ylabel('btp','fontsize',10)
grid on;
hold off
figure(2)
plot(error2)
title('误差','fontsize',10,'fontangle','normal')


figure(3)
bf=error2./output_test;
plot(100*bf,'r.-')
%title('误差百分比','fontsize',10,'fontangle','normal')
xlabel('样本序列','fontsize',10)
ylabel('误差（%）','fontweight','bold')
grid on;
%% 结果分析

