%% Generate and predict sunspot sequence (12-step prediction)
clear; clc;
%% Load data
load('sunspot_12step_train_test_data_normalized.mat');

rng(2025);  % Set random seed for reproducibility
%% Parameter optimization settings (adjust ranges as needed)
param_ranges = {
    {'Nr',  'integer',    [10, 300]},      % Reservoir size
    {'Nl',  'integer',    [2, 15]},         % Number of network layers
    {'spectral_radius', 'continuous', [0.1, 0.9]}, % Spectral radius
    {'leaking_rate',    'continuous', [0.1, 1]}, % Leaking rate
    {'compression_ratio', 'continuous', [0.1, 0.9]}, % Compression ratio
    {'inter_scaling', 'continuous', [0.1, 0.9]},    % Inter-layer scaling factor
    {'bias', 'continuous', [0.1, 3]}                % Bias term
};

n_particles = 20;     % Number of particles in PSO
max_iter = 10;        % Maximum number of optimization iterations

%% Start parallel computing pool
if isempty(gcp('nocreate'))
    parpool('Processes', 8);  % Initialize 8 parallel workers
end

%% PSO optimization core function
function [best_params, best_mse] = pso_optimize_deepcsesn(param_ranges, data, target, train_range, test_range, n_particles, max_iter)
    % Initialize particle swarm with random parameters
    particles = initialize_particles(param_ranges, n_particles);
    
    % Initialize global best tracking variables
    global_best_mse = Inf;
    global_best_params = [];
    
    % Create progress bar for optimization
    h = waitbar(0, 'PSO optimization progress...');
    
    for iter = 1:max_iter
        parfor i = 1:n_particles  % Parallel evaluation of particles
            % Decode particle position to model parameters
            params = decode_particle(particles(i), param_ranges);
            
            % Evaluate model performance
            current_mse = evaluate_deepcsesn(params, data, target, train_range, test_range);
            
            % Update individual best performance
            if current_mse < particles(i).best_mse
                particles(i).best_mse = current_mse;
                particles(i).best_position = particles(i).position;
            end
        end
        
        % Update global best parameters
        [current_best_mse, idx] = min([particles.best_mse]);
        if current_best_mse < global_best_mse
            global_best_mse = current_best_mse;
            global_best_params = decode_particle(particles(idx), param_ranges);
        end
        
        % Update particle velocities and positions
        particles = update_particles(particles, global_best_params, param_ranges);
        
        % Update progress bar
        waitbar(iter/max_iter, h, sprintf('Iteration %d/%d, Current best MSE: %.4f', iter, max_iter, global_best_mse));
    end
    close(h);  % Close progress bar
    
    best_params = global_best_params;
    best_mse = global_best_mse;
end

%% Helper functions
function particles = initialize_particles(param_ranges, n_particles)
    % Define particle structure template
    template = struct(...
        'position', [], ...         % Current parameter values
        'velocity', [], ...         % Current velocity vector
        'best_position', [], ...    % Best position found
        'best_mse', Inf ...         % Best MSE achieved
    );
    
    % Preallocate particle array
    particles = repmat(template, 1, n_particles);
    
    % Initialize each particle
    for i = 1:n_particles
        % Initialize position within parameter ranges
        position = zeros(1, length(param_ranges));
        for p = 1:length(param_ranges)
            param = param_ranges{p};
            switch param{2}
                case 'integer'
                    position(p) = randi(param{3});  % Random integer in range
                case 'continuous'
                    position(p) = param{3}(1) + rand() * (param{3}(2) - param{3}(1));  % Random float in range
            end
        end
        
        % Assign initial values
        particles(i).position = position;
        particles(i).velocity = zeros(1, length(param_ranges));
        particles(i).best_position = position;  % Initial best position
        particles(i).best_mse = Inf;
    end
end

function params = decode_particle(particle, param_ranges)
    % Convert particle position to parameter structure
    params = struct();
    for p = 1:length(param_ranges)
        param = param_ranges{p};
        params.(param{1}) = particle.position(p);
    end
end

function particles = update_particles(particles, global_best, param_ranges)
    % PSO parameters for velocity update
    inertia_weight = 0.5;
    cognitive_weight = 1.5;  % Weight for personal best
    social_weight = 1.5;     % Weight for global best
    
    for i = 1:length(particles)
        % Velocity update calculations
        r1 = rand(1, length(param_ranges));  % Random factors for cognitive component
        r2 = rand(1, length(param_ranges));  % Random factors for social component
        
        % Construct global best parameter vector (matches param_ranges order)
        global_best_vector = [...
            global_best.Nr, ...
            global_best.Nl, ...
            global_best.spectral_radius, ...
            global_best.leaking_rate, ...
            global_best.compression_ratio, ...
            global_best.inter_scaling, ...
            global_best.bias, ...
        ]; 
        
        % Update velocity using PSO formula
        particles(i).velocity = inertia_weight * particles(i).velocity + ...
            cognitive_weight * r1 .* (particles(i).best_position - particles(i).position) + ...
            social_weight * r2 .* (global_best_vector - particles(i).position);
        
        % Update position
        particles(i).position = particles(i).position + particles(i).velocity;
        
        % Enforce parameter boundaries
        for p = 1:length(param_ranges)
            param = param_ranges{p};
            % Handle integer parameters
            if strcmp(param{2}, 'integer')
                particles(i).position(p) = round(particles(i).position(p));
            end
            % Apply boundary constraints
            particles(i).position(p) = max(particles(i).position(p), param{3}(1));
            particles(i).position(p) = min(particles(i).position(p), param{3}(2));
        end
    end
end


%% Model evaluation function
function mse = evaluate_deepcsesn(params, data, target, train_range, test_range)
    % Initialize ICSDESN network
    net = ICSDESN();
    net.Nu = 1;  % Input dimension set to 1 for sunspot data
    net.Nr = params.Nr;
    net.Nl = params.Nl;
    net.spectral_radius = params.spectral_radius;
    net.leaking_rate = params.leaking_rate;
    net.compression_ratio = params.compression_ratio;
    net.inter_scaling = params.inter_scaling;
    net.bias = params.bias;
    net.washout = 200;  % Washout period to stabilize dynamics
    net.readout_regularization = 1e-11;  % Regularization parameter
    net.initialize();

    
    % Training phase
    states = net.run(data(:, train_range(1:end-1)));
    net.train_readout(target(:, train_range(2:end)));
    
    % Testing phase
    test_states = net.run(data(:, test_range(1:end-1)));
    test_output = net.compute_output(test_states, true);
    target_valid = target(:, test_range(net.washout + 2:end));
    
    % Calculate mean squared error
    mse = mean((target_valid - test_output).^2, 'all');
end

% Helper function to calculate R-squared (coefficient of determination)
function r2 = calculate_r2(predicted, actual)
    ss_total = sum((actual - mean(actual, 'all')).^2, 'all');  % Total sum of squares
    ss_residual = sum((actual - predicted).^2, 'all');         % Residual sum of squares
    
    % Avoid division by zero for constant data
    if ss_total < 1e-10
        r2 = 0;
    else
        r2 = 1 - (ss_residual / ss_total);  % R-squared formula
    end
end

%% Run parameter optimization
[best_params, best_mse] = pso_optimize_deepcsesn(...
    param_ranges, input_data, target, train_range, test_range, n_particles, max_iter);

disp('===== Optimization Results =====');
disp(best_params);
disp(['Best Test MSE: ', num2str(best_mse)]);


%% Train final model with optimized parameters
net_final = ICSDESN();
net_final.Nu = 1;  
net_final.Nr = best_params.Nr;
net_final.Nl = best_params.Nl;
net_final.spectral_radius = best_params.spectral_radius;
net_final.leaking_rate = best_params.leaking_rate;
net_final.compression_ratio = best_params.compression_ratio;
net_final.inter_scaling = best_params.inter_scaling;
net_final.bias = best_params.bias;
net_final.washout = 200;
net_final.readout_regularization = 1e-11;
net_final.initialize();


% Training phase with timing
tic;
train_states = net_final.run(input_data(:, train_range(1:end-1)));
net_final.train_readout(target(:, train_range(2:end)));
train_output = net_final.compute_output(train_states, true);
train_time = toc;

% Testing phase with timing
tic;
test_states = net_final.run(input_data(:, test_range(1:end-1)));
test_output = net_final.compute_output(test_states, true);
target_test = target(:, test_range(net_final.washout + 2:end));
test_time = toc;


% Calculate performance metrics
error = target_test - test_output;           % Error vector
abs_error = abs(error);                      % Absolute error
squared_error = error.^2;                    % Squared error
epsilon = 1e-6;                              % Avoid division by zero
percentage_error = abs_error ./ (abs(target_test) + epsilon); % Percentage error

% Core metric calculations
final_mse = mean(squared_error);             % Mean Squared Error
final_rmse = sqrt(final_mse);                % Root Mean Squared Error
final_mae = mean(abs_error);                 % Mean Absolute Error
final_mape = mean(percentage_error) * 100;   % Mean Absolute Percentage Error
final_r2 = calculate_r2(test_output, target_test);  % R-squared value

% Variance calculations for metrics
mse_var = var(squared_error(:));             % Variance of MSE components
mae_var = var(abs_error(:));                 % Variance of MAE components
mape_var = var(percentage_error(:)) * 100^2; % Variance of MAPE components (%²)

% Display performance results
disp('===== Model Performance Metrics =====');
disp(['Test MSE:    ', num2str(final_mse, '%.4e'), ' (Variance: ', num2str(mse_var, '%.4e'), ')']);
disp(['Test RMSE:   ', num2str(final_rmse, '%.4e')]);
disp(['Test MAE:    ', num2str(final_mae, '%.4e'), ' (Variance: ', num2str(mae_var, '%.4e'), ')']);
disp(['Test MAPE:   ', num2str(final_mape, '%.2f'), '% (Variance: ', num2str(mape_var, '%.2f'), '%²)']);
disp(['Test R²:     ', num2str(final_r2, '%.6f')]);  % R-squared value
disp(['Training Time:   ', num2str(train_time, '%.4f'), ' seconds']);
disp(['Testing Time:   ', num2str(test_time, '%.4f'), ' seconds']);



%% Statistical evaluation with 100 independent runs
num_runs = 100;  % Number of iterations for statistical significance

% Preallocate storage for metrics including R-squared
mse_list = zeros(1, num_runs);
rmse_list = zeros(1, num_runs);
mae_list = zeros(1, num_runs);
mape_list = zeros(1, num_runs);
r2_list = zeros(1, num_runs);  % Store R-squared values

% Initialize progress bar
h = waitbar(0, 'Running model... 0% complete');

for i = 1:num_runs
    % Model initialization with optimized parameters
    net_final = ICSDESN();
    net_final.Nu = 1;  
    net_final.Nr = best_params.Nr;
    net_final.Nl = best_params.Nl;
    net_final.spectral_radius = best_params.spectral_radius;
    net_final.leaking_rate = best_params.leaking_rate;
    net_final.compression_ratio = best_params.compression_ratio;
    net_final.inter_scaling = best_params.inter_scaling;
    net_final.bias = best_params.bias;
    net_final.washout = 200;
    net_final.readout_regularization = 1e-11;
    net_final.initialize();

    % Training
    train_states = net_final.run(input_data(:, train_range(1:end-1)));
    net_final.train_readout(target(:, train_range(2:end)));
    
    % Testing
    test_states = net_final.run(input_data(:, test_range(1:end-1)));
    test_output = net_final.compute_output(test_states, true);
    target_test = target(:, test_range(net_final.washout + 2:end));
    
    % Calculate metrics for current run
    error = target_test - test_output;
    abs_error = abs(error);
    squared_error = error.^2;
    percentage_error = abs_error ./ (abs(target_test) + 1e-6);
    
    % Calculate and store R-squared
    current_r2 = calculate_r2(test_output, target_test);
    
    % Store results
    mse_list(i) = mean(squared_error(:));
    rmse_list(i) = sqrt(mse_list(i));
    mae_list(i) = mean(abs_error(:));
    mape_list(i) = mean(percentage_error(:)) * 100;
    r2_list(i) = current_r2;  % Store R-squared value
    
    % Update progress bar
    waitbar(i/num_runs, h, sprintf('Running model... %.1f%% complete', i/num_runs*100));
end
close(h);


% Calculate average values across all runs
avg_mse = mean(mse_list);
avg_rmse = mean(rmse_list);
avg_mae = mean(mae_list);
avg_mape = mean(mape_list);
avg_r2 = mean(r2_list);  % Average R-squared

% Calculate variance of metrics across runs
var_mse = var(mse_list);
var_rmse = var(rmse_list);
var_mae = var(mae_list);
var_mape = var(mape_list);
var_r2 = var(r2_list);  % Variance of R-squared values

% Display statistical results
disp('===== 100 Runs Statistical Results =====');
fprintf('Average Test MSE:  %.6e (Variance: %.6e)\n', avg_mse, var_mse);
fprintf('Average Test RMSE: %.6e (Variance: %.6e)\n', avg_rmse, var_rmse);
fprintf('Average Test MAE:  %.6e (Variance: %.6e)\n', avg_mae, var_mae);
fprintf('Average Test MAPE: %.6f%% (Variance: %.6f%%)\n', avg_mape, var_mape);
fprintf('Average Test R²:   %.6f (Variance: %.6f)\n', avg_r2, var_r2);  % R-squared statistics


