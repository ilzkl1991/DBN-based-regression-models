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
%ѡ����������������ݹ�һ��
inputn=mapminmax('apply',input_train,inputps);
outputn=mapminmax('apply',output_train,outputps);
% 
% data_new=[inputn',outputn'];

inputn_test=mapminmax('apply',input_test,inputps);
outputn_test=mapminmax('apply',output_test,outputps);
datatest=[inputn_test',outputn_test'];

 %% K��ֵ
% [Idx, Center]=kmeans(data_new,3,'dist','sqEuclidean','rep',4);
% data1=data_new(Idx==1,:)';
% data2=data_new(Idx==2,:)';
% data3=data_new(Idx==3,:)';
load center center
%% ��ʱѧϰ
[m,~]=size(datatest);
[n,~]=size(center);
for i=1:m
    for j=1:n
         dist(i,j)=sqrt(sum(datatest(i,:)-center(j,:)).^2); 
         angle(i,j)=datatest(i,:)*center(j,:)'/(sqrt(sum(datatest(i,:)).^2)*sqrt(sum(center(j,:)).^2));
         S(i,j)=0.5*sqrt(exp(-dist(i,j).^2))+0.5*angle(i,j);
    end
     [Y(i),I(i)] = min(dist(i,:)); %Y��ʾ��Сŷ������,I��ʾ��Ӧ���кţ���������һ���


end
BB=[datatest';I]; 

for i=1:m
    if BB(end,i)==1
        an(:,i)=rbf1(BB(1:9,i));
    else
       an(:,i)=rbf2(BB(1:9,i));
    end  
end
BPoutput=an;
error2=BPoutput-output_test;

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
goal = 0.1;                % ѵ������ƽ����(Ĭ��Ϊ0) 
spread = 10;                % ��ֵԽ��,��Ҫ����Ԫ��Խ��(Ĭ��Ϊ1) 
 MN = size(xn_train,2);% �����Ԫ��(Ĭ��Ϊѵ����������) 
% MN=35;
DF = 1;                     % ��ʾ���(Ĭ��Ϊ25) 
net = newrb(P,T,goal,spread,MN,DF); 

case 3 
     
P = xn_train; 
T = dn_train; 
spread = 0.5;                % ��ֵԽ��,��Ҫ����Ԫ��Խ��(Ĭ��Ϊ1) 
net = newgrnn(P,T,spread); 
     
end

%% BP����Ԥ��
% %Ԥ�����ݹ�һ��
% inputn_test=mapminmax('apply',input_test,inputps);
 
%����Ԥ�����

an=sim(net,BB(1:9,:));
% an=sim(net,inputn);
 
%�����������һ��
BPoutput2=mapminmax('reverse',an,outputps);
error3=BPoutput2-output_test;

MSE1=sum((BPoutput-output_test).^2)/length(output_test);
MSE2=sum((BPoutput2-output_test).^2)/length(output_test);
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
%�������

figure(4)
plot(BPoutput2,'r-*')
hold on
%title('ʵ��ֵ��Ԥ��ֵ���ͼ','fontsize',10,'fontangle','normal')
plot(output_test,'b.-')
legend('Ԥ��ֵ','ʵ��ֵ','Location','NorthEast');
xlabel('��������','fontsize',10)
ylabel('btp','fontsize',10)
grid on;
hold off
figure(5)
plot(error3)
title('���','fontsize',10,'fontangle','normal')

figure(6)
bf2=error3./output_test;
plot(100*bf2,'r.-')
%title('���ٷֱ�','fontsize',10,'fontangle','normal')
xlabel('��������','fontsize',10)
ylabel('��%��','fontweight','bold')
grid on;
%�������