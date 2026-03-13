% MAE 247: ICF Robustness to Inaccurate Knowledge of N (Experiment 9)
clear; clc; close all;

% Start parallel pool if one doesn't exist
if isempty(gcp('nocreate'))
    parpool; 
end

%% Experiment Parameters
N = 31;                 
K = 100;                
Delta = 2;              
SR = 300;               
t_max = 40;             
epsilon = 0.65 / Delta; 
gridsize = 500;

delta_N_vec = -30:2:30; 
num_d = length(delta_N_vec);

num_envs = 10;           
num_tracks = 10;         

H = [1 0 0 0; 0 1 0 0];
Phi = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];
Q = diag([10, 10, 1, 1]); 
R = 100 * eye(2);
B = inv(R);

%% Communication Graph Setup
Adj = zeros(N, N);
for d = 1:(Delta/2)
    Adj = Adj + diag(ones(N-d, 1), d) + diag(ones(N-d, 1), -d);
    Adj = Adj + diag(ones(d, 1), N-d) + diag(ones(d, 1), -(N-d));
end

%% Main PARALLEL Monte Carlo Loop
% pre-allocate arrays to hold the results from each parallel worker
env_ckf_err  = zeros(num_envs, 1);     env_ckf_sq   = zeros(num_envs, 1);
env_kcf_err  = zeros(num_envs, 1);     env_kcf_sq   = zeros(num_envs, 1);
env_gkcf_err = zeros(num_envs, 1);     env_gkcf_sq  = zeros(num_envs, 1);
env_icf_err  = zeros(num_envs, num_d); env_icf_sq   = zeros(num_envs, num_d);
env_dp       = zeros(num_envs, 1);

fprintf('Starting Parallel Monte Carlo Simulation...\n');

parfor env = 1:num_envs
    % accumulators for each specific CPU core
    loc_ckf_err = 0;  loc_ckf_sq = 0;
    loc_kcf_err = 0;  loc_kcf_sq = 0;
    loc_gkcf_err = 0; loc_gkcf_sq = 0;
    loc_icf_err = zeros(1, num_d);
    loc_icf_sq  = zeros(1, num_d);
    loc_dp = 0;
    
    rng(env * 1000);
    sensor_pos = gridsize * rand(2, N);
    sensor_theta = 2 * pi * rand(1, N);
    
    for trk = 1:num_tracks
        rng(env * 1000 + trk);
        % target
        x_true_init = [gridsize/2; gridsize/2; 2*cos(rand*2*pi); 2*sin(rand*2*pi)]; 
        target_track = zeros(4, t_max);
        current_x = x_true_init;
        for t = 1:t_max
            current_x = Phi * current_x + chol(Q)' * randn(4,1);
            target_track(:, t) = current_x;
        end
        
        % init shared prios
        P_prior_init = diag([100, 100, 10, 10]);
        J_prior_init = inv(P_prior_init);
        
        x_prior_base = zeros(4, N);
        J_prior_base = repmat(J_prior_init, 1, 1, N);
        shared_prior_noise = chol(P_prior_init)' * randn(4,1);
        for i = 1:N
            x_prior_base(:, i) = x_true_init + shared_prior_noise; 
        end
        
        x_prior_ckf = x_prior_base(:, 1); J_prior_ckf = J_prior_init;
        x_prior_kcf = x_prior_base;       J_prior_kcf = J_prior_base;
        x_prior_gkcf = x_prior_base;      J_prior_gkcf = J_prior_base;
        
        % separate prior tracker for each Delta N scenario, icf
        x_prior_icf_all = zeros(4, N, num_d);
        J_prior_icf_all = zeros(4, 4, N, num_d);
        for d_idx = 1:num_d
            x_prior_icf_all(:,:,d_idx) = x_prior_base;
            J_prior_icf_all(:,:,:,d_idx) = J_prior_base;
        end
        for t = 1:t_max
            x_true = target_track(:, t);
            
            % observation
            U_all = zeros(4, 4, N); u_all = zeros(4, N);
            for i = 1:N
                vec_to_target = x_true(1:2) - sensor_pos(:, i);
                dist = norm(vec_to_target);
                angle_diff = wrapToPi(atan2(vec_to_target(2), vec_to_target(1)) - sensor_theta(i));
                
                if dist <= SR && abs(angle_diff) <= pi/6
                    Z_i = H * x_true + chol(R)' * randn(2,1); 
                    U_all(:,:,i) = H' * B * H;
                    u_all(:,i)   = H' * B * Z_i;
                end
            end
            
            S = zeros(4, 4, N); y = zeros(4, N);
            for i = 1:N
                inclusive_neighborhood = [find(Adj(i, :)), i];
                for j = inclusive_neighborhood
                    S(:, :, i) = S(:, :, i) + U_all(:,:,j);
                    y(:, i)   = y(:, i)   + u_all(:,j);
                end
            end
            
            % ckf
            U_total = sum(U_all, 3); u_total = sum(u_all, 2);
            J_post_ckf = J_prior_ckf + U_total;
            x_post_ckf = J_post_ckf \ (J_prior_ckf * x_prior_ckf + u_total);
            
            err_ckf = norm(x_true(1:2) - x_post_ckf(1:2));
            loc_ckf_err = loc_ckf_err + (N * err_ckf);
            loc_ckf_sq  = loc_ckf_sq  + (N * err_ckf^2);
            
            J_prior_ckf = inv(Phi * inv(J_post_ckf) * Phi' + Q);
            x_prior_ckf = Phi * x_post_ckf;
            % kcf
            x_kcf_k = x_prior_kcf;
            J_post_kcf = zeros(4, 4, N);
            for i = 1:N; J_post_kcf(:,:,i) = J_prior_kcf(:,:,i) + S(:,:,i); end
            for k_iter = 1:K
                x_new = x_kcf_k;
                for i = 1:N
                    neighbors = find(Adj(i, :));
                    sum_x_diff = sum(x_kcf_k(:, neighbors), 2) - length(neighbors)*x_kcf_k(:, i);
                    J_inv = inv(J_prior_kcf(:,:,i));
                    consensus_term = (epsilon / (1 + norm(J_inv))) * J_inv * sum_x_diff;
                    meas_term = (k_iter==1) * (inv(J_post_kcf(:,:,i)) * (y(:,i) - S(:,:,i)*x_prior_kcf(:,i)));
                    x_new(:, i) = x_kcf_k(:, i) + meas_term + consensus_term;
                end
                x_kcf_k = x_new;
            end
            for i = 1:N
                err_kcf = norm(x_true(1:2) - x_kcf_k(1:2, i));
                loc_kcf_err = loc_kcf_err + err_kcf;
                loc_kcf_sq  = loc_kcf_sq  + err_kcf^2;
                J_prior_kcf(:,:,i) = inv(Phi * inv(J_post_kcf(:,:,i)) * Phi' + Q);
                x_prior_kcf(:, i) = Phi * x_kcf_k(:, i);
            end
            
            % gkcf
            x_gkcf_k = x_prior_gkcf;
            J_post_gkcf = zeros(4, 4, N);
            for i = 1:N
                neighbors = find(Adj(i, :));
                sum_J_diff = sum(J_prior_gkcf(:,:,neighbors), 3) - length(neighbors)*J_prior_gkcf(:,:,i);
                J_post_gkcf(:,:,i) = J_prior_gkcf(:,:,i) + epsilon * sum_J_diff + S(:,:,i);
            end
            for k_iter = 1:K
                x_new = x_gkcf_k;
                for i = 1:N
                    neighbors = find(Adj(i, :));
                    sum_Jx_diff = zeros(4, 1);
                    for j = neighbors
                        sum_Jx_diff = sum_Jx_diff + J_prior_gkcf(:,:,j) * (x_gkcf_k(:, j) - x_gkcf_k(:, i));
                    end
                    consensus_term = inv(J_post_gkcf(:,:,i)) * epsilon * sum_Jx_diff;
                    meas_term = (k_iter==1) * (inv(J_post_gkcf(:,:,i)) * (y(:,i) - S(:,:,i)*x_prior_gkcf(:,i)));
                    x_new(:, i) = x_gkcf_k(:, i) + meas_term + consensus_term;
                end
                x_gkcf_k = x_new;
            end
            for i = 1:N
                err_gkcf = norm(x_true(1:2) - x_gkcf_k(1:2, i));
                loc_gkcf_err = loc_gkcf_err + err_gkcf;
                loc_gkcf_sq  = loc_gkcf_sq  + err_gkcf^2;
                J_prior_gkcf(:,:,i) = inv(Phi * inv(J_post_gkcf(:,:,i)) * Phi' + Q);
                x_prior_gkcf(:, i) = Phi * x_gkcf_k(:, i);
            end
            
            % icf, sweep for n
            for d_idx = 1:num_d
                N_hat = N + delta_N_vec(d_idx);
                if N_hat <= 0; N_hat = 1e-5; end % Prevent div by zero
                
                % Extract the prior state specifically for this Delta N track
                x_prior_icf = x_prior_icf_all(:,:,d_idx);
                J_prior_icf = J_prior_icf_all(:,:,:,d_idx);
                
                V = zeros(4, 4, N); v = zeros(4, N);
                for i = 1:N
                    V(:, :, i) = (1/N_hat) * J_prior_icf(:, :, i) + U_all(:,:,i);
                    v(:, i)    = (1/N_hat) * J_prior_icf(:, :, i) * x_prior_icf(:, i) + u_all(:,i);
                end
                
                for k_iter = 1:K
                    V_new = V; v_new = v;
                    for i = 1:N
                        neighbors = find(Adj(i, :));
                        sum_V_diff = sum(V(:,:,neighbors), 3) - length(neighbors)*V(:,:,i);
                        sum_v_diff = sum(v(:,neighbors), 2) - length(neighbors)*v(:,i);
                        V_new(:, :, i) = V(:, :, i) + epsilon * sum_V_diff;
                        v_new(:, i)   = v(:, i)   + epsilon * sum_v_diff;
                    end
                    V = V_new; v = v_new;
                end
                
                for i = 1:N
                    J_post_icf = N_hat * V(:, :, i); 
                    x_post_icf = V(:, :, i) \ v(:, i); 
                    
                    err_icf = norm(x_true(1:2) - x_post_icf(1:2));
                    loc_icf_err(d_idx) = loc_icf_err(d_idx) + err_icf;
                    loc_icf_sq(d_idx)  = loc_icf_sq(d_idx)  + err_icf^2;
                    
                    J_prior_icf(:, :, i) = inv(Phi * inv(J_post_icf) * Phi' + Q);
                    x_prior_icf(:, i)    = Phi * x_post_icf;
                end
                
                % save to branch tracker
                x_prior_icf_all(:,:,d_idx) = x_prior_icf;
                J_prior_icf_all(:,:,:,d_idx) = J_prior_icf;
            end
            
            loc_dp = loc_dp + N;
        end  
    end  
    
    % store local to global
    env_ckf_err(env) = loc_ckf_err;     env_ckf_sq(env) = loc_ckf_sq;
    env_kcf_err(env) = loc_kcf_err;     env_kcf_sq(env) = loc_kcf_sq;
    env_gkcf_err(env) = loc_gkcf_err;   env_gkcf_sq(env) = loc_gkcf_sq;
    env_icf_err(env, :) = loc_icf_err;  env_icf_sq(env, :) = loc_icf_sq;
    env_dp(env) = loc_dp;
end

%%  merge parallel process resutls
fprintf('Aggregating results...\n');
total_dp = sum(env_dp);

ckf_mean = sum(env_ckf_err) / total_dp;
ckf_std  = sqrt((sum(env_ckf_sq) / total_dp) - ckf_mean^2);

kcf_mean = sum(env_kcf_err) / total_dp;
kcf_std  = sqrt((sum(env_kcf_sq) / total_dp) - kcf_mean^2);

gkcf_mean = sum(env_gkcf_err) / total_dp;
gkcf_std  = sqrt((sum(env_gkcf_sq) / total_dp) - gkcf_mean^2);

mean_error_icf_vec = sum(env_icf_err, 1) / total_dp;
std_icf_vec = sqrt((sum(env_icf_sq, 1) / total_dp) - mean_error_icf_vec.^2);

%% Plotting Figure 17
figure('Name', 'Figure 17: Robustness to N');
hold on;

% Create constant arrays for the baseline algorithms to stretch across the graph
ckf_line  = repmat(ckf_mean, 1, num_d);
kcf_line  = repmat(kcf_mean, 1, num_d);
gkcf_line = repmat(gkcf_mean, 1, num_d);

% CKF
plot(delta_N_vec, ckf_line, '-go', 'LineWidth', 1.5, 'MarkerIndices', 1:5:num_d); 
plot(delta_N_vec, ckf_line + 0.2*ckf_std, ':g', 'LineWidth', 1);
plot(delta_N_vec, ckf_line - 0.2*ckf_std, ':g', 'LineWidth', 1);

% KCF
plot(delta_N_vec, kcf_line, '-kd', 'LineWidth', 1.5, 'MarkerIndices', 1:5:num_d);
plot(delta_N_vec, kcf_line + 0.2*kcf_std, ':k', 'LineWidth', 1);
plot(delta_N_vec, kcf_line - 0.2*kcf_std, ':k', 'LineWidth', 1);

% GKCF
plot(delta_N_vec, gkcf_line, '-bs', 'LineWidth', 1.5, 'MarkerIndices', 1:5:num_d);
plot(delta_N_vec, gkcf_line + 0.2*gkcf_std, ':b', 'LineWidth', 1);
plot(delta_N_vec, gkcf_line - 0.2*gkcf_std, ':b', 'LineWidth', 1);

% ICF
plot(delta_N_vec, mean_error_icf_vec, '-rx', 'LineWidth', 1.5); 
plot(delta_N_vec, mean_error_icf_vec + 0.2*std_icf_vec, ':r', 'LineWidth', 1);
plot(delta_N_vec, mean_error_icf_vec - 0.2*std_icf_vec, ':r', 'LineWidth', 1);

xlabel('Deviation from actual N, \Delta N', 'Interpreter', 'tex');
ylabel('Mean Error, e and S.D. \pm 0.2\sigma', 'Interpreter', 'tex');
title('Experiment 9: Robustness to Error in N (Actual N = 31)');
legend('CKF', '', '', 'KCF', '', '', 'GKCF', '', '', 'ICF', 'Location', 'northeast');
grid on;
xlim([-30 30]);
ylim([4 22]);
hold off;