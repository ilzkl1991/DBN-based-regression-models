%% 清空环境变量
clc
clear

%% 训练数据预测数据提取及归一化

%找出训练数据和预测数据
A=xlsread('12.xlsx');
input_train1=A(2:350,1:9);
output_train1=A(2:350,10);

input_test1=A(351:400,1:9);
output_test1=A(351:400,10);
input_train=input_train1';
input_test=input_test1';
output_train=output_train1';
output_test=output_test1';

%选连样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP网络训练
% %初始化网络结构
net=newelm(minmax(inputn),[25,1],{'tansig','purelin'});
net.trainparam.show=100;%每迭代100次显示1次
net.trainparam.epochs=1000;%最大迭代次数2000
net.trainparam.goal=0.1;%迭代目标
net=init(net);%初始化网络
%网络训练
[net,tr]=train(net,inputn,outputn);
% save elman1 net;     %保存训练好的网络模型

%% BP网络预测
%预测数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
 
%网络预测输出

an=sim(net,inputn_test);
% an=sim(net,inputn);
 
%网络输出反归一化
BPoutput=mapminmax('reverse',an,outputps);

error2=BPoutput-output_test;
MSE1=sum((BPoutput-output_test).^2)/length(BPoutput);
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
