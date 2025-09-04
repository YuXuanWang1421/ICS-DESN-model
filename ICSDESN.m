classdef ICSDESN < DeepESN
    % CompressedDeepESN: 实现层间压缩感知的深度回声网络
    % 继承自DeepESN，在层间加入高斯测量矩阵压缩信号
    
    properties
        M          % 压缩后的维度（每层，根据压缩率计算）
        Phi        % 高斯测量矩阵集合（cell数组，Phi{l}对应第l层到l+1层的压缩矩阵）
        compression_ratio    % 压缩率
        Psi        % 稀疏基矩阵（例如傅里叶变换、小波变换、DCT等）
        noise_std  % 高斯噪声的标准差（默认0.01）
    end
    
    methods
        function self = ICSDESN()
            % 构造函数，继承父类并初始化压缩参数
            self@DeepESN();  % 调用父类构造函数
            self.compression_ratio = 0.8; % 设置默认压缩率
            % M在initialize方法中根据compression_ratio和Nr计算
            self.noise_std = 0;        % 设置默认噪声标准差
        end
        
        function initialize(self)
            % 重写初始化方法，生成压缩矩阵和调整层间权重
            
            % 计算压缩后的维度M（取整）
            self.M = round(self.Nr * self.compression_ratio);
            
            % 生成高斯压缩矩阵（层间）
            self.Phi = cell(self.Nl-1, 1);
            for l = 1:self.Nl-1
                self.Phi{l} = randn(self.M, self.Nr) * (1/sqrt(self.M)); % 能量守恒
            end
            
            % 初始化稀疏基矩阵（例如DCT矩阵）
            self.Psi = dctmtx(self.Nr);  % 使用DCT作为稀疏基
            
            % 初始化输入矩阵（父类逻辑）
            self.Win = 2*rand(self.Nr, self.Nu+1)-1;
            switch self.input_scaling_mode
                case 'bynorm'
                    self.Win = self.input_scaling * self.Win / norm(self.Win);
                case 'byrange'
                    self.Win = self.Win * self.input_scaling;
            end
            
            % 初始化层间权重（调整维度为压缩后尺寸）
            for l = 2:self.Nl
                % 输入维度变为 M+1 (压缩后维度 + 偏置)
                self.Wil{l} = 2*rand(self.Nr, self.M+1)-1; 
                switch self.input_scaling_mode
                    case 'bynorm'
                        self.Wil{l} = self.inter_scaling * self.Wil{l} / norm(self.Wil{l});
                    case 'byrange'
                        self.Wil{l} = self.Wil{l} * self.inter_scaling;
                end
            end
            
            % 初始化循环权重（与父类相同）
            for l = 1:self.Nl
                self.W{l} = spfun(@(x) 2*x - 1, sprandn(self.Nr, self.Nr, 0.1));
                I = eye(self.Nr);
                Wt = (1-self.leaking_rate)*I + self.leaking_rate*self.W{l};
                Wt = self.spectral_radius * Wt / max(abs(eig(Wt)));
                self.W{l} = (Wt - (1-self.leaking_rate)*I)/self.leaking_rate;
            end
            
            % 初始化状态
            self.initial_state = zeros(self.Nr,1);
            self.init_state();
        end
        
        function states = run(self, input)
            % 重写运行方法，在层间加入压缩操作
            
            Nt = size(input,2);
            for l = 1:self.Nl
                self.run_states{l} = zeros(self.Nr, Nt);
            end
            
            old_state = self.state;
            for t = 1:Nt
                for l = 1:self.Nl
                    x = old_state{l};
                    
                    % 处理层间输入（压缩）
                    if l == 1
                        u = input(:,t);
                        input_part = self.Win * [u; self.bias];
                    else
                        % 压缩前一层输出
                        compressed = self.Phi{l-1} *  self.run_states{l-1}(:,t);
                        input_part = self.Wil{l} * [compressed; self.bias];
                    end
                    
                    % 状态更新
                    noise = self.noise_std * randn(size(x));  % 生成高斯噪声
                    self.state{l} = (1 - self.leaking_rate)*x + ...
                                   self.leaking_rate * self.f(input_part + self.W{l}*x);
                    self.run_states{l}(:,t) = self.state{l};
                    old_state{l} = self.state{l};
                end
            end
            states = self.run_states;
        end
    end
end