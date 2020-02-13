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

s1=40;
net=newff(inputdata11,outputdata11,s1);
net.trainParam.epochs=100;
net.trainParam.lr=1;
net.trainParam.goal=0.03;
net.trainParam.max_fail = 200; 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
net=train(net,inputdata11,outputdata11);

%% Training for ELMAN
% net=newelm(minmax(inputn),[40,1],{'tansig','tansig'});
% net.trainparam.show=100;
% net.trainparam.epochs=3000;
% net.trainparam.goal=0.03;
% net=init(net);
% [net,tr]=train(net,inputn,outputn);

%% Training for RBF
% switch 2
% case 1 
% spread = 1;               
% net = newrb(inputn,outputn);    
% % save BRPRBF net;
% case 2 
% P = train_x2; 
% T = train_y2;  
% goal = 0.013;               
% spread =1;              
% MN = size(P,2);
% DF = 1;                     
% net = newrb(inputn,outputn,goal,spread,MN,DF); 
%    
%     case 3     
% P = Ytrain1; 
% T = train_y1;  
% spread = 1;                
% net = newgrnn(inputn,outputn,spread); 
%      
% end

%% Testing the model
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

