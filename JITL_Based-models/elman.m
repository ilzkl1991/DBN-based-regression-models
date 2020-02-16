%% ��ջ�������
clc
clear

%% ѵ������Ԥ��������ȡ����һ��

%�ҳ�ѵ�����ݺ�Ԥ������
A=xlsread('12.xlsx');
input_train1=A(2:350,1:9);
output_train1=A(2:350,10);

input_test1=A(351:400,1:9);
output_test1=A(351:400,10);
input_train=input_train1';
input_test=input_test1';
output_train=output_train1';
output_test=output_test1';

%ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP����ѵ��
% %��ʼ������ṹ
net=newelm(minmax(inputn),[25,1],{'tansig','purelin'});
net.trainparam.show=100;%ÿ����100����ʾ1��
net.trainparam.epochs=1000;%����������2000
net.trainparam.goal=0.1;%����Ŀ��
net=init(net);%��ʼ������
%����ѵ��
[net,tr]=train(net,inputn,outputn);
% save elman1 net;     %����ѵ���õ�����ģ��

%% BP����Ԥ��
%Ԥ�����ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
 
%����Ԥ�����

an=sim(net,inputn_test);
% an=sim(net,inputn);
 
%�����������һ��
BPoutput=mapminmax('reverse',an,outputps);

error2=BPoutput-output_test;
MSE1=sum((BPoutput-output_test).^2)/length(BPoutput);
figure(1)
plot(BPoutput,'r-*')
hold on
%title('ʵ��ֵ��Ԥ��ֵ���ͼ','fontsize',10,'fontangle','normal')
plot(output_test,'b.-')
legend('Ԥ��ֵ','ʵ��ֵ','Location','NorthEast');
xlabel('��������','fontsize',10)
ylabel('btp','fontsize',10)
grid on;
hold off
figure(2)
plot(error2)
title('���','fontsize',10,'fontangle','normal')


figure(3)
bf=error2./output_test;
plot(100*bf,'r.-')
%title('���ٷֱ�','fontsize',10,'fontangle','normal')
xlabel('��������','fontsize',10)
ylabel('��%��','fontweight','bold')
grid on;
%% �������
