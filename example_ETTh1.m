clc;clear;
rng(2025);
%% Load ETTh1 data and perform preprocessing
T = readtable('ETTh1.csv');

% Separate feature columns (input) and target column (OT, prediction target)
feature_cols = T(:, {'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL','OT'}); % 7 feature columns
target_col = T.OT; % OT column as prediction target

% Convert to numerical matrices
features = table2array(feature_cols);
targets = target_col;

% Min-max normalization
min_vals = min(features);
max_vals = max(features);
features_normalized = (features - min_vals) ./ (max_vals - min_vals);

min_target = min(targets);
max_target = max(targets);
targets_normalized = (targets - min_target) ./ (max_target - min_target);

%% Sliding window parameter settings
window_size = 24;    % Sliding window size
stride = 1;          % Sliding step size

%% Convert time series data to sliding window format
function [windowed_input, windowed_target] = create_sliding_windows(input_series, target_series, window_size, stride)
    num_samples = size(input_series, 1);
    num_windows = floor((num_samples - window_size) / stride);
    num_features = size(input_series, 2);
    
    % Preallocate memory - Dimension: (Number of features × Window size × Number of windows)
    windowed_input = zeros(num_features, window_size, num_windows);
    windowed_target = zeros(1, num_windows);
    
    for i = 1:num_windows
        start_idx = (i-1)*stride + 1;
        end_idx = start_idx + window_size - 1;
        
        % Extract window data - Note dimension conversion
        window_data = input_series(start_idx:end_idx, :)';  % Transpose to (Number of features × Window size)
        windowed_input(:, :, i) = window_data;
        
        % Target value is the next value after the window
        windowed_target(i) = target_series(end_idx + 1);
    end
end

%% Apply sliding window processing
[windowed_input, windowed_target] = create_sliding_windows(features_normalized, targets_normalized, window_size, stride);

% Check data dimensions
fprintf('Windowed input data dimensions: %d × %d × %d\n', size(windowed_input));
fprintf('Windowed target data dimensions: %d × %d\n', size(windowed_target));

%% Reorganize data to meet new format requirements
% New format: (Number of features × Window size) × Number of windows
% Concatenate 24 time steps (each containing 7 features) as input features

% Calculate dimension sizes
num_features = size(windowed_input, 1);    % 7 features
window_size = size(windowed_input, 2);     % 24 time steps
num_windows = size(windowed_input, 3);     % 17396 windows

% Reshape input data: 
% 1. First adjust dimension order to (Number of features × Window size × Number of windows)
% 2. Then reshape to (Number of features × Window size) × Number of windows
input_data_reshaped = reshape(windowed_input, num_features * window_size, num_windows);

% Keep target data consistent with number of windows
target_data = windowed_target;

% Check reshaped dimensions
fprintf('Reshaped input data dimensions: %d × %d\n', size(input_data_reshaped));
fprintf('Target data dimensions: %d × %d\n', size(target_data));

%% Split into training and testing sets
total_samples = num_windows;  % Total samples equal to number of windows
train_ratio = 0.8;
train_samples = floor(total_samples * train_ratio);

train_range = 1:train_samples;
test_range = (train_samples + 1):total_samples;

% Note: We use sample indices instead of time step indices here
input_data = input_data_reshaped;
target = target_data;

% Split training and testing sets by window
train_input = input_data_reshaped(:, train_range);
train_target = target_data(:, train_range);
test_input = input_data_reshaped(:, test_range);
test_target = target_data(:, test_range);

% Check dimensions after splitting
fprintf('Training input data dimensions: %d × %d\n', size(train_input));
fprintf('Training target data dimensions: %d × %d\n', size(train_target));
fprintf('Testing input data dimensions: %d × %d\n', size(test_input));
fprintf('Testing target data dimensions: %d × %d\n', size(test_target));

%% Parameter optimization settings
param_ranges = {
    {'Nr',  'integer',    [50, 300]},      % Reservoir size
    {'Nl',  'integer',    [2, 15]},         % Number of network layers
    {'spectral_radius', 'continuous', [0.5, 1]}, % Spectral radius
    {'leaking_rate',    'continuous', [0.1, 0.9]}, % Leaking rate
    {'compression_ratio', 'continuous', [0.1, 0.9]} % Compression ratio
};

n_particles = 20;     % Number of particles
max_iter = 10;       % Maximum number of iterations

%% Start parallel pool
if isempty(gcp('nocreate'))
    parpool('Processes', 24);  % Use 24 worker processes
end

%% PSO optimization core function
function [best_params, best_mse] = pso_optimize_deepcsesn(param_ranges, data, target, train_range, test_range, n_particles, max_iter)
    % Initialize particle swarm
    particles = initialize_particles(param_ranges, n_particles);
    
    % Global best parameters
    global_best_mse = Inf;
    global_best_params = [];
    
    % Progress bar
    h = waitbar(0,'PSO optimization progress...');
    
     % Initialize timing
    iterTimes = zeros(max_iter,1);
    
    for iter = 1:max_iter
        parfor i = 1:n_particles  % Use parfor for parallel computation
            % Parameter decoding
            params = decode_particle(particles(i), param_ranges);
            
            % Model evaluation
            current_mse = evaluate_deepcsesn(params, data, target, train_range, test_range);
            
            % Update individual best
            if current_mse < particles(i).best_mse
                particles(i).best_mse = current_mse;
                particles(i).best_position = particles(i).position;
            end
        end
        
        % Update global best
        [current_best_mse, idx] = min([particles.best_mse]);
        if current_best_mse < global_best_mse
            global_best_mse = current_best_mse;
            global_best_params = decode_particle(particles(idx), param_ranges);
        end
        
        % Update particle states
        particles = update_particles(particles, global_best_params, param_ranges);
        
        % Update progress bar
        waitbar(iter/max_iter, h, sprintf('Iteration %d/%d, Current best MSE: %.4f', iter, max_iter, global_best_mse));
    end
    close(h);
    best_params = global_best_params;
    best_mse = global_best_mse;
end

%% Helper functions
function particles = initialize_particles(param_ranges, n_particles)
    % Predefine structure template
    template = struct(...
        'position', [], ...
        'velocity', [], ...
        'best_position', [], ...
        'best_mse', Inf ...
    );
    
    % Preallocate structure array
    particles = repmat(template, 1, n_particles);
    
    % Initialize each particle
    for i = 1:n_particles
        % Initialize position
        position = zeros(1, length(param_ranges));
        for p = 1:length(param_ranges)
            param = param_ranges{p};
            switch param{2}
                case 'integer'
                    position(p) = randi(param{3});
                case 'continuous'
                    position(p) = param{3}(1) + rand()*(param{3}(2)-param{3}(1));
            end
        end
        
        % Assign structure fields uniformly
        particles(i).position = position;
        particles(i).velocity = zeros(1, length(param_ranges));
        particles(i).best_position = position; % Initial best position is current position
        particles(i).best_mse = Inf;
    end
end

function params = decode_particle(particle, param_ranges)
    params = struct();
    for p = 1:length(param_ranges)
        param = param_ranges{p};
        params.(param{1}) = particle.position(p);
    end
end

function particles = update_particles(particles, global_best, param_ranges)
    inertia_weight = 0.5;
    cognitive_weight = 1.5;
    social_weight = 1.5;
    
    for i = 1:length(particles)
        % Velocity update --------------------------------------------------------
        r1 = rand(1, length(param_ranges));
        r2 = rand(1, length(param_ranges));
        
        % Key modification: Construct global best parameter vector (must exactly match the order of param_ranges)
        global_best_vector = [...
            global_best.Nr, ...
            global_best.Nl, ...
            global_best.spectral_radius, ...
            global_best.leaking_rate, ...
            global_best.compression_ratio, ...
        ]; 
        
        particles(i).velocity = inertia_weight * particles(i).velocity + ...
            cognitive_weight * r1 .* (particles(i).best_position - particles(i).position) + ...
            social_weight * r2 .* (global_best_vector - particles(i).position);
        
        % Position update --------------------------------------------------------
        particles(i).position = particles(i).position + particles(i).velocity;
        
        % Boundary handling --------------------------------------------------------
        for p = 1:length(param_ranges)
            param = param_ranges{p};
            % Handle integer parameters
            if strcmp(param{2}, 'integer')
                particles(i).position(p) = round(particles(i).position(p));
            end
            % Boundary constraints
            particles(i).position(p) = max(particles(i).position(p), param{3}(1));
            particles(i).position(p) = min(particles(i).position(p), param{3}(2));
        end
    end
end

%% Model evaluation function
function mse = evaluate_deepcsesn(params, data, target, train_range, test_range)
    % Network initialization
    net = ICSDESN();
    net.Nu = size(data, 1);  % Use number of features as input dimension
    net.Nr = params.Nr;
    net.Nl = params.Nl;
    net.spectral_radius = params.spectral_radius;
    net.leaking_rate = params.leaking_rate;
    net.compression_ratio = params.compression_ratio;
    net.washout = 1000;  % Dynamically adjust washout
    net.readout_regularization = 1e-11;
    net.initialize();

    % Training phase
    states = net.run(data(:, train_range(1:end-1)));
    net.train_readout(target(:, train_range(2:end)));
    
    % Testing phase
    test_states = net.run(data(:, test_range(1:end-1)));
    test_output = net.compute_output(test_states, true);
    target_valid = target(:, test_range(net.washout+2:end));
    
    % Calculate MSE
    mse = mean((target_valid - test_output).^2, 'all');
end

% Function to calculate R-squared (coefficient of determination)
function r2 = calculate_r2(predicted, actual)
    ss_total = sum((actual - mean(actual)).^2);
    ss_residual = sum((actual - predicted).^2);
    r2 = 1 - (ss_residual / ss_total);
end

%% Run optimization

fprintf('\n=== Starting PSO Optimization ===\n');
optimizationStart = tic;  % PSO optimization timing start

[best_params, best_mse] = pso_optimize_deepcsesn(...
    param_ranges, input_data, target, train_range, test_range, n_particles, max_iter);

optimizationTime = toc(optimizationStart);  % PSO optimization timing end
fprintf('PSO optimization time: %.2f minutes\n', optimizationTime/60);

disp('===== Optimization Results =====');
disp(best_params);
disp(['Best Test MSE: ', num2str(best_mse)]);

%% Train final model with optimized parameters
net_final = ICSDESN();
net_final.Nu = size(input_data, 1);  % Use number of features as input dimension
net_final.Nr = best_params.Nr;
net_final.Nl = best_params.Nl;
net_final.spectral_radius = best_params.spectral_radius;
net_final.leaking_rate = best_params.leaking_rate;
net_final.compression_ratio = best_params.compression_ratio;
net_final.washout = 1000;  % Dynamically adjust washout
net_final.readout_regularization = 1e-11;
net_final.initialize();

% Training
train_tic = tic;
train_states = net_final.run(input_data(:, train_range(1:end-1)));
net_final.train_readout(target(:, train_range(2:end)));
train_time = toc(train_tic);

train_output = net_final.compute_output(train_states, true);  % Save training output

% Testing
test_tic = tic;
test_states = net_final.run(input_data(:, test_range(1:end-1)));
test_output = net_final.compute_output(test_states, true);
target_test = target(:, test_range(net_final.washout+2:end));
test_time = toc(test_tic);


% Calculate basic error terms
error = target_test - test_output;           % Error matrix
abs_error = abs(error);                       % Absolute error
squared_error = error.^2;                     % Squared error
epsilon = 1e-6; 
percentage_error = abs_error ./ (abs(target_test) + epsilon); % Percentage error

% Calculate core metrics
final_mse = mean(squared_error, 'all');
final_rmse = sqrt(final_mse);
final_mae = mean(abs_error, 'all');
final_mape = mean(percentage_error, 'all') * 100;
final_r2 = calculate_r2(test_output, target_test);  % Calculate R-squared

% Calculate variance metrics
mse_var = var(squared_error(:));              % Variance of MSE components
mae_var = var(abs_error(:));                  % Variance of MAE components
mape_var = var(percentage_error(:)) * 100^2;   % Variance of MAPE components (unit: %^2)

disp('===== Final Model Performance =====');
disp(['Test MSE:     ', num2str(final_mse, '%.4e'), ' (Variance: ', num2str(mse_var, '%.4e'), ')']);
disp(['Test RMSE:    ', num2str(final_rmse, '%.4e')]);
disp(['Test MAE:     ', num2str(final_mae, '%.4e'), ' (Variance: ', num2str(mae_var, '%.4e'), ')']);
disp(['Test MAPE:    ', num2str(final_mape, '%.2f'), '% (Variance: ', num2str(mape_var, '%.2f'), '%²)']);
disp(['Test R²:      ', num2str(final_r2, '%.4f')]);  % Display R-squared
disp(['Training time:    ', num2str(train_time, '%.3f'), ' seconds']);
disp(['Testing time:    ', num2str(test_time, '%.3f'), ' seconds']);

%% Train final model with optimized parameters (100 iterations)
num_runs = 100;  % Define number of iterations

% Preallocate storage matrices
mse_list = zeros(1, num_runs);
rmse_list = zeros(1, num_runs);
mae_list = zeros(1, num_runs);
mape_list = zeros(1, num_runs);
r2_list = zeros(1, num_runs);  % Store R-squared values

% Initialize progress bar
h = waitbar(0, 'Running model... 0% complete');

for i = 1:num_runs
    % Model initialization
    net_final = ICSDESN();
    net_final.Nu = size(input_data, 1);  % Use number of features as input dimension
    net_final.Nr = best_params.Nr;
    net_final.Nl = best_params.Nl;
    net_final.spectral_radius = best_params.spectral_radius;
    net_final.leaking_rate = best_params.leaking_rate;
    net_final.compression_ratio = best_params.compression_ratio;
    net_final.washout = min(50, floor(size(input_data, 2)*0.1));  % Dynamically adjust washout
    net_final.readout_regularization = 1e-11;
    net_final.initialize();

    % Training
    train_states = net_final.run(input_data(:, train_range(1:end-1)));
    net_final.train_readout(target(:, train_range(2:end)));
    
    % Testing
    test_states = net_final.run(input_data(:, test_range(1:end-1)));
    test_output = net_final.compute_output(test_states, true);
    target_test = target(:, test_range(net_final.washout+2:end));
    
    % Calculate metrics
    error = target_test - test_output;
    abs_error = abs(error);
    squared_error = error.^2;
    percentage_error = abs_error ./ (abs(target_test) + 1e-6);
    
    % Store results
    mse_list(i) = mean(squared_error(:));
    rmse_list(i) = sqrt(mse_list(i));
    mae_list(i) = mean(abs_error(:));
    mape_list(i) = mean(percentage_error(:)) * 100;
    r2_list(i) = calculate_r2(test_output, target_test);  % Calculate and store R-squared
    
    % Update progress bar
    waitbar(i/num_runs, h, sprintf('Running model... %.1f%% complete', i/num_runs*100));
end
close(h);

% Average values
avg_mse = mean(mse_list);
avg_rmse = mean(rmse_list);
avg_mae = mean(mae_list);
avg_mape = mean(mape_list);
avg_r2 = mean(r2_list);  % Average R-squared

% Variance calculations
var_mse = var(mse_list);
var_rmse = var(rmse_list);
var_mae = var(mae_list);
var_mape = var(mape_list);
var_r2 = var(r2_list);  % Variance of R-squared

disp('===== 100 Runs Statistical Results =====');
fprintf('Average Test MSE:  %.6e (Variance: %.6e)\n', avg_mse, var_mse);
fprintf('Average Test RMSE: %.6e (Variance: %.6e)\n', avg_rmse, var_rmse);
fprintf('Average Test MAE:  %.6e (Variance: %.6e)\n', avg_mae, var_mae);
fprintf('Average Test MAPE: %.6f%% (Variance: %.6f%%)\n', avg_mape, var_mape);
fprintf('Average Test R²:   %.6f (Variance: %.6f)\n', avg_r2, var_r2);  % Display R-squared statistics

%% Plot results
figure;
plot(test_target(50:end),'r');
hold on;
plot(test_output,'b');
legend('Actual Values', 'Predicted Values');
title('Comparison of Actual and Predicted Values');
xlabel('Time Step');
ylabel('Value');
grid on;

