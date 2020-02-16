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

%  ex1 train a 100 hidden unit RBM and visualize its weights
numcases=80;
numdims=size(P,2);
numbatches=10;
% 训练数据
for i=1:numbatches
    train1=P((i-1)*numcases+1:i*numcases,:);
    batchdata(:,:,i)=train1;
end%将分好的10组数据都放在batchdata中



%% 2.训练RBM
%% rbm参数
maxepoch=100;%训练rbm的次数
numhid=30; numpen=30; numpen2=30; numpen3=30;%dbn隐含层的节点数
disp('构建一个4层的深度置信网络DBN用于特征提取');
%% 无监督预训练
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d ',numdims,numhid);
restart=1;
rbm1;%使用cd-k训练rbm，注意此rbm的可视层不是二值的，而隐含层是二值的
vishid1=vishid;hidrecbiases=hidbiases;


fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d ',numhid,numpen);
batchdata=batchposhidprobs;%将第一个RBM的隐含层的输出作为第二个RBM 的输入
numhid=numpen;%将numpen的值赋给numhid，作为第二个rbm隐含层的节点数
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d ',numpen,numpen2);%200-100
batchdata=batchposhidprobs;%显然，将第二哥RBM的输出作为第三个RBM的输入
numhid=numpen2;%第三个隐含层的节点数
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
% 
% fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d\n ',numpen2,numpen3);%200-100
% batchdata=batchposhidprobs;%显然，将第二哥RBM的输出作为第三个RBM的输入
% numhid=numpen3;%第三个隐含层的节点数
% restart=1;
% rbm1;
% hidpen3=vishid; penrecbiases3=hidbiases; hidgenbiases3=visbiases;


%%%% 将预训练好的RBM用于初始化DBN权重%%%%%%%%%
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];
% w4=[hidpen3; penrecbiases3];

%% 有监督回归层训练
%===========================训练过程=====================================%
%==========DBN无监督用于提取特征，需要加上有监督的回归层==================%
%由于含有偏执，所以实际数据应该包含一列全为1的数，即w0x0+w1x1+..+wnxn 其中x0为1的向量 w0为偏置b
N1 = size(P,1);
digitdata = [P ones(N1,1)];

w1probs = 1./(1 + exp(-digitdata*w1));
w1probs = [w1probs  ones(N1,1)];

w2probs = 1./(1 + exp(-w1probs*w2));
w2probs = [w2probs  ones(N1,1)];

w3probs = 1./(1 + exp(-w2probs*w3));
H= w3probs'; %DBN的输出  也是BP的输入
%% BP网络训练
inputn=H(1:30,1:150);
outputn=T';
% %初始化网络结构
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
% 训练误差
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
% P = train_x2; 
% T = train_y2;  
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

input_test1=A(751:800,1:6);     %测试数据(仅测试一行)
output_test1=A(751:800,7);
input_test=input_test1';
output_test=output_test1';
test_x=mapminmax('apply',input_test,inputps,0,1)'; 

N1 = size(test_x,1);
digitdata1 = [test_x ones(N1,1)];

w1probs1 = 1./(1 + exp(-digitdata1*w1));
w1probs1 = [w1probs1  ones(N1,1)];

w2probs1 = 1./(1 + exp(-w1probs1*w2));
w2probs1 = [w2probs1  ones(N1,1)];

w3probs1 = 1./(1 + exp(-w2probs1*w3)); 
test_x1 =w3probs1';
% 
% w4probs1 = 1./(1 + exp(-w3probs1*w4)); 
% test_x1 = w4probs1';

an=sim(net,test_x1);
%修改部分，输出预测值

BPoutput=mapminmax('reverse',an,outputps);
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

