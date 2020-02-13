function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;             %n=3����
    m = size(x, 1);       %mΪѵ�����ݵ�������һ����ѵ��������MSE����
    
    x = [ones(m,1) x];    %ones(m,1)����m*1��ȫ1��
    nn.a{1} = x;          %��ֱ����ṹ�������Ԫ����˴�Ϊ��nn.a�ĳ�ʼ����nn.a{1}Ϊn*2�ľ���  %a1Ϊ����㣬ֱ��Ϊ�������룬Ϊ�����²�������׼��

    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function 
            case 'sigm'
%                 nn.a{i - 1}
%                  nn.W{i - 1}'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');   %�˴�Ϊ�����������sigmoid��Ϊ�����
%                case 'linear'
%                 nn.a{i} = nn.a{i - 1} * nn.W{i - 1}';
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term  ���ƫ����
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    switch nn.output   %����㼤�������
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');  %���һ������ڵ�����ֵ
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
%             nn.L = sum(nn.e .^ 2)/ m 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end
