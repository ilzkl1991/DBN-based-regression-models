clear all;
clc
close all;

A=xlsread('train1.xlsx');

input_train1=A(1:(end-50),1:9);
output_train1=A(1:(end-50),10);

input_test1=A((end-50):end,1:9);
output_test1=A((end-50):end,10);


input_train=input_train1';
input_test=input_test1';
output_train=output_train1';
output_test=output_test1';
 load('outputps.mat')
 load('inputps.mat')
%选连样本输入输出数据归一化
inputn=mapminmax('apply',input_train,inputps);
outputn=mapminmax('apply',output_train,outputps);
% 
% data_new=[inputn',outputn'];

inputn_test=mapminmax('apply',input_test,inputps);
outputn_test=mapminmax('apply',output_test,outputps);
datatest=[inputn_test',outputn_test'];

%  %% K均值
% [Idx, Center]=kmeans(data_new,3,'dist','sqEuclidean','rep',4);
% data1=data_new(Idx==1,:)';
% data2=data_new(Idx==2,:)';
% data3=data_new(Idx==3,:)';
load center center
%% 即时学习
[m,~]=size(datatest);
[n,~]=size(center);
for i=1:m
    for j=1:n
         dist(i,j)=sqrt(sum(datatest(i,:)-center(j,:)).^2); 
         angle(i,j)=datatest(i,:)*center(j,:)'/(sqrt(sum(datatest(i,:)).^2)*sqrt(sum(center(j,:)).^2));
         S(i,j)=0.5*sqrt(exp(-dist(i,j).^2))+0.5*angle(i,j);
    end
     [Y(i),I(i)] = min(dist(i,:)); %Y表示最小欧拉距离,I表示对应的行号，即所属哪一类别


end
BB=[datatest';I]; 

for i=1:m
    if BB(end,i)==1
        an(:,i)=Elman1(BB(1:9,i));
    else
       an(:,i)=Elman2(BB(1:9,i));
    end  
end
BPoutput=an;
error2=BPoutput-output_test;
net=newelm(minmax(inputn),[20,1],{'tansig','purelin'});
%net=newff(inputn,outputn, [5,12,4],{'tansig','tansig','tansig','purelin'},'trainlm','learngdm','mse');


net.trainparam.show=100;%每迭代100次显示1次
net.trainparam.epochs=1000;%最大迭代次数2000
net.trainparam.goal=0.06;%迭代目标
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
BPoutput2=mapminmax('reverse',an,outputps);
error3=BPoutput2-output_test;
% 
MSE1=sum((BPoutput-output_test).^2)/length(output_test);
MSE2=sum((BPoutput2-output_test).^2)/length(output_test);
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
% figure(2)
% plot(error2)
% title('误差','fontsize',10,'fontangle','normal')

figure(3)
bf=error2./output_test;
plot(100*bf,'r.-')
%title('误差百分比','fontsize',10,'fontangle','normal')
xlabel('样本序列','fontsize',10)
ylabel('误差（%）','fontweight','bold')
grid on;
%结果分析

figure(4)
plot(BPoutput2,'r-*')
hold on
%title('实际值与预测值拟合图','fontsize',10,'fontangle','normal')
plot(output_test,'b.-')
legend('预测值','实际值','Location','NorthEast');
xlabel('样本序列','fontsize',10)
ylabel('btp','fontsize',10)
grid on;
hold off
% figure(5)
% plot(error3)
% title('误差','fontsize',10,'fontangle','normal')

figure(6)
bf2=error3./output_test;
plot(100*bf2,'r.-')
%title('误差百分比','fontsize',10,'fontangle','normal')
xlabel('样本序列','fontsize',10)
ylabel('误差（%）','fontweight','bold')
grid on;
%结果分析