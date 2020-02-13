%web browser http://www.ilovematlab.cn/thread-60357-1-1.html
%% 双隐含层BP神经网络
%% 清空环境变量
clc
clear

%% 训练数据预测数据提取及归一化

%找出训练数据和预测数据
A=xlsread('12.xlsx');
[~,n]=size(A);
rowrank = randperm(size(A, 1)); 
B = A(rowrank, :);
input_train1=B(2:350,1:9);
output_train1=B(2:350,10);

input_test1=B(351:400,1:9);
output_test1=B(351:400,10);
input_train=input_train1';
input_test=input_test1';
output_train=output_train1';
output_test=output_test1';

%选连样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP网络训练
net=newff(minmax(inputn),[25,1],{'tansig','purelin'},'trainlm');
net.trainParam.epochs=600;
net.trainParam.lr=0.1;%(0.01-0.8)
net.trainParam.goal=0.06;
net.trainParam.max_fail = 10;
net.trainParam.show = 10; %显示次数
net.trainParam.showCommandLine = 1;
net.trainParam.time = inf;
net.trainParam.min_grad = 1e-6;
%网络训练
net=train(net,inputn,outputn);
save BRPBP net;     %保存训练好的网络模型

Wjk=net.IW{1,1}; 
Wij=net.LW{2,1}; 
B1=net.b{1}; 
B2=net.b{2}; 
%% BP网络预测
%预测数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
 
%网络预测输出

an=sim(net,inputn_test);
% an=sim(net,inputn);
 
%网络输出反归一化
BPoutput=mapminmax('reverse',an,outputps);

error2=BPoutput-output_test;

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
