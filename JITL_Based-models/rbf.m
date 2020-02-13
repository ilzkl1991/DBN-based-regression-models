%% ��ջ�������
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

%ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
 xn_train=inputn;
 dn_train=outputn;
switch 2
case 1 
         
% ��Ԫ����ѵ���������� 
P = inputn; 
T = outputn; 
spread = 0.5;                % ��ֵԽ��,���ǵĺ���ֵ�ʹ�(Ĭ��Ϊ1) 
net = newrb(P,T); 
% save BRPRBF net;
case 2 
     
% ��Ԫ��������,������ѵ���������� 
P = xn_train; 
T = dn_train; 
goal = 0.15;                % ѵ������ƽ����(Ĭ��Ϊ0) 
spread = 10;                % ��ֵԽ��,��Ҫ����Ԫ��Խ��(Ĭ��Ϊ1) 
%  MN = size(xn_train,2);% �����Ԫ��(Ĭ��Ϊѵ����������) 
MN=40;
DF = 1;                     % ��ʾ���(Ĭ��Ϊ25) 
net = newrb(P,T,goal,spread,MN,DF); 
% save RBF1 net;
case 3 
     
P = xn_train; 
T = dn_train; 
spread = 0.5;                % ��ֵԽ��,��Ҫ����Ԫ��Խ��(Ĭ��Ϊ1) 
net = newgrnn(P,T,spread); 
     
end

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
