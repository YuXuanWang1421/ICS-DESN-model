classdef CS_ESN < handle
    properties
        Nr      % 储备池维度
        Nu      % 输入维度
        Ny      % 输出维度
        M       % 压缩后维度
        Phi     % 压缩矩阵(M x Nr)
        compression_rate % 压缩率 (0-1)

        spectral_radius
        input_scaling      
        leaking_rate       
        input_scaling_mode 
        f                  
        bias               
        
        washout            
        readout_regularization 
        
        Win     % 输入->储备池权重矩阵
        W       % 储备池循环权重矩阵
        Wout    % 输出权重矩阵
        
        run_states     % 运行时原始状态矩阵(Nr x Nt)
        initial_state  
        state          
    end
    
    methods
        function self = CS_ESN()
            self.default();
        end
        
        function self = default(self)
            self.Nr = 100;
            self.Nu = 1;
            self.Ny = 1;
            self.M = [];  % 初始化时不计算
            self.compression_rate = 0.5;  % 默认压缩率
            
            self.spectral_radius = 0.9;
            self.input_scaling = 1;
            self.leaking_rate = 0.3;
            self.input_scaling_mode = 'bynorm';
            self.f = @tanh;
            self.bias = 1;
            
            self.washout = 100;
            self.readout_regularization = 1e-6;
            
            self.Win = [];
            self.W = [];
            self.Wout = [];
            self.Phi = [];
            
            self.initial_state = [];
            self.state = [];
        end
        
        function init_state(self)
            self.state = self.initial_state;
            self.run_states = [];
        end
        
        function initialize(self)

            self.M = max(1, round(self.Nr * self.compression_rate));

            % 初始化输入权重
            self.Win = 2*rand(self.Nr, self.Nu + 1) - 1;
            switch self.input_scaling_mode
                case 'bynorm'
                    self.Win = self.input_scaling * self.Win / norm(self.Win);
                case 'byrange'
                    self.Win = self.Win * self.input_scaling;
            end
            
            % 初始化循环权重并调整谱半径
            self.W = sprand(self.Nr, self.Nr,  0.01);
            max_eig = max(abs(eig(full(self.W))));
            self.W = self.W * (self.spectral_radius / max_eig);
            
            % 初始化压缩矩阵(高斯随机矩阵)
            self.Phi = randn(self.M, self.Nr) / sqrt(self.Nr);
            
            % 初始化状态
            self.initial_state = zeros(self.Nr, 1);
            self.init_state();
        end

        function set.compression_rate(self, rate)
            % 属性验证
            if rate <= 0 || rate > 1
                error('压缩率必须在(0,1]范围内')
            end
            self.compression_rate = rate;
        end

        function states = run(self, input)
            Nt = size(input, 2);
            self.run_states = zeros(self.Nr, Nt);
            
            for t = 1:Nt
                u = input(:, t);
                x = self.state;
                
                % 状态更新方程
                input_part = self.Win * [u; self.bias];
                reservoir_input = input_part + self.W * x;
                
                self.state = (1 - self.leaking_rate) * x + ...
                            self.leaking_rate * self.f(reservoir_input);
                
                self.run_states(:, t) = self.state;
            end
            states = self.run_states;
        end
        
        function train_readout(self, target)
            % 对状态进行压缩感知
            raw_states = self.run_states(:, self.washout+1:end);
            compressed_states = self.Phi * raw_states;  % 压缩操作
            
            X = [compressed_states; self.bias * ones(1, size(compressed_states,2))];
            Y = target(:, self.washout+1:end);
            
            % 岭回归训练
            self.Wout = Y * X' / (X*X' + self.readout_regularization*eye(size(X,1)));
            self.Ny = size(self.Wout, 1);
        end
        
        function output = compute_output(self, states, remove_washout)
            if remove_washout
                states = states(:, self.washout+1:end);
            end
            % 对状态进行压缩
            compressed_states = self.Phi * states;
            output = self.Wout * [compressed_states; self.bias * ones(1, size(states,2))];
        end
        
        function [outputTR, outputTS] = train_test(self, input, target, train_idx, test_idx)
            self.init_state();
            
            % 训练阶段
            train_input = input(:, train_idx);
            train_target = target(:, train_idx);
            states = self.run(train_input);
            self.train_readout(train_target);
            outputTR = self.compute_output(states, true);
            
            % 测试阶段
            test_input = input(:, test_idx);
            test_states = self.run(test_input);
            outputTS = self.compute_output(test_states, false);
        end
    end
end