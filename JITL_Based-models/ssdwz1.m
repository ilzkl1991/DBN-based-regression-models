%web browser http://www.ilovematlab.cn/thread-60357-1-1.html
%% ˫������BP������
%% ��ջ�������
clc
clear

%% ѵ������Ԥ��������ȡ����һ��

%�ҳ�ѵ�����ݺ�Ԥ������
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

%ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP����ѵ��
net=newff(minmax(inputn),[25,1],{'tansig','purelin'},'trainlm');
net.trainParam.epochs=600;
net.trainParam.lr=0.1;%(0.01-0.8)
net.trainParam.goal=0.06;
net.trainParam.max_fail = 10;
net.trainParam.show = 10; %��ʾ����
net.trainParam.showCommandLine = 1;
net.trainParam.time = inf;
net.trainParam.min_grad = 1e-6;
%����ѵ��
net=train(net,inputn,outputn);
save BRPBP net;     %����ѵ���õ�����ģ��

Wjk=net.IW{1,1}; 
Wij=net.LW{2,1}; 
B1=net.b{1}; 
B2=net.b{2}; 
%% BP����Ԥ��
%Ԥ�����ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
 
%����Ԥ�����

an=sim(net,inputn_test);
% an=sim(net,inputn);
 
%�����������һ��
BPoutput=mapminmax('reverse',an,outputps);

error2=BPoutput-output_test;

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
