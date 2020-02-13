clc
clear

A=xlsread('BB.xlsx'); %The data for traininig and testing

inputdata=A(1:150,1:6);  %Training input data
outputdata=A(1:150,7); 

inputdata1=inputdata';
outputdata1=outputdata';

%normalize
[inputdata11,inputps]=mapminmax(inputdata1,0,1);

[outputdata11,outputps]=mapminmax(outputdata1,0,1);


%% BP网络训练
%初始化网络结构

switch 2
case 1 
         
P = inputdata11; 
T = outputdata11;  
spread = 1;                % 此值越大,覆盖的函数值就大(默认为1) 
net = newrb(P,T);    
% save BRPRBF net;
case 2 
     
% 神经元数逐步增加,最多就是训练样本个数 
P = inputdata11; 
T = outputdata11; 
goal = 0.013;                % 训练误差的平方和(默认为0) 
spread =1;                % 此值越大,需要的神经元就越少(默认为1) 
MN = size(P,2);% 最大神经元数(默认为训练样本个数) 
DF = 1;                     % 显示间隔(默认为25) 
net = newrb(P,T,goal,spread,MN,DF); 
   
    case 3     
P = Ytrain1; 
T = train_y1;  
spread = 1;                % 此值越大,需要的神经元就越少(默认为1) 
net = newgrnn(P,T,spread); 
     
end
input_test1=A(751:800,1:6);   
output_test1=A(751:800,7);
input_test=input_test1';
output_test=output_test1';
test_x=mapminmax('apply',input_test,inputps,0,1)'; 

an=sim(net,test_x'); %prediction result for testing


BPoutput=mapminmax('reverse',an,outputps);
error2=BPoutput-output_test;
% MSE1=sum((BPoutput-output_test).^2)/length(BPoutput);
figure(1)
plot(BPoutput,'r-*')
hold on
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




