%% 清空环境变量
clc
clear

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
 xn_train=inputn;
 dn_train=outputn;
switch 2
case 1 
         
% 神经元数是训练样本个数 
P = inputn; 
T = outputn; 
spread = 0.5;                % 此值越大,覆盖的函数值就大(默认为1) 
net = newrb(P,T); 
% save BRPRBF net;
case 2 
     
% 神经元数逐步增加,最多就是训练样本个数 
P = xn_train; 
T = dn_train; 
goal = 0.15;                % 训练误差的平方和(默认为0) 
spread = 10;                % 此值越大,需要的神经元就越少(默认为1) 
%  MN = size(xn_train,2);% 最大神经元数(默认为训练样本个数) 
MN=40;
DF = 1;                     % 显示间隔(默认为25) 
net = newrb(P,T,goal,spread,MN,DF); 
% save RBF1 net;
case 3 
     
P = xn_train; 
T = dn_train; 
spread = 0.5;                % 此值越大,需要的神经元就越少(默认为1) 
net = newgrnn(P,T,spread); 
     
end

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
