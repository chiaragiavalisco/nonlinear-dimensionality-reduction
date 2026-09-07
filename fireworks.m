% =========================================================================
% Title:       Nonlinear Dimensionality Reduction: 2D Fireworks Dataset
% Course:      Numerical Methods for Data Mining
% Author:      Chiara Giavalisco
% Description: Evaluates MDS, Kernel PCA (RBF kernel), and Isomap on a 
%              2D multimodal dataset composed of 3 synthetic Gaussian clusters
%              (Fireworks)
% =========================================================================

close all; clear; clc;

% --- Data Generation & Parameters ---
n_clusters = 3;             % Number of "rays" or clusters
points_per_cluster = 500;   % Number of points per cluster
spread = 0.3;               % Spread or variance within each cluster
radius = 1.2;               % Distance from the origin for the clusters

X = [];
Y = [];
labels = [];

for i = 1:n_clusters
    % Angle for the cluster's direction
    theta = 2 * pi * (i / n_clusters);
    
    % Center of the cluster
    x_center = radius * cos(theta);
    y_center = radius * sin(theta);
    
    % Generate points with Gaussian spread around the center
    x = x_center + spread * randn(points_per_cluster, 1);
    y = y_center + spread * randn(points_per_cluster, 1);
    
    % Append to the dataset
    X = [X; x];
    Y = [Y; y];
    labels = [labels; ones(points_per_cluster, 1) * i]; % Assign cluster label
end

% Combine the data into a matrix for shuffling
data = [X, Y, labels];
shuffled_indices = randperm(size(data, 1));
data = data(shuffled_indices, :);

% Extract shuffled points and labels
X_shuffled = data(:, 1);
Y_shuffled = data(:, 2);
labels_shuffled = data(:, 3);

% --- Visualization: Fireworks Ground Truth ---
figure(1);
gscatter(X_shuffled, Y_shuffled, labels_shuffled);
title('2D Fireworks');
xlabel('X');
ylabel('Y');
axis equal;

% --- Method 1: Multidimensional Scaling (MDS) ---
M = points_per_cluster;
distances = pdist(data);
k = 3;                      % Target reduced dimension
D = squareform(distances);
[n, ~] = size(D);           % Number of data points

% Double centering operation
I = eye(n);
O = ones(n);
H = I - ((1/n)*O);
B = -0.5 * H * D * H';

[U, L, W] = svd(B);
v = zeros(n, k);
s = zeros(k, k);
for j = 1:k
    v(:,j) = W(:,j);        % Extracted principal dimensions
    for i = 1:k
        s(i,j) = sqrt(L(i,j));
    end
end

z1 = s * v';                % Projected data in the k dimension
z1 = z1';

% Error calculation with MDS
errMDS = norm((data'*data) - (z1'*z1));
errrelMDS = errMDS / norm(data'*data);
disp('MDS error='); disp(errrelMDS);

% MDS Plots
zero = zeros(M, 1);

figure(2);
hold on;
gscatter(z1(1:M, 2), zero, labels_shuffled(1:M));
gscatter(z1(M+1:2*M, 2), zero, labels_shuffled(M+1:2*M));
gscatter(z1(2*M+1:end, 2), zero, labels_shuffled(2*M+1:end));
hold off;
xlabel('MD1');
title('1D Multidimensional Scaling');

figure(3);
hold on;
gscatter(z1(1:M, 1), z1(1:M, 2), labels_shuffled(1:M));
gscatter(z1(M+1:2*M, 1), z1(M+1:2*M, 2), labels_shuffled(M+1:2*M));
gscatter(z1(2*M+1:end, 1), z1(2*M+1:end, 2), labels_shuffled(2*M+1:end));
hold off;
xlabel('MD1');
ylabel('MD2');
title('2D Multidimensional Scaling');

% --- Method 2: Kernel PCA with Gaussian (RBF) Kernel ---
X = data;
sigma = 0.4;                % Hyperparameter for 2D embedding
sigma1 = 0.8;               % Hyperparameter for 1D embedding
gamma = 0.01;               % RBF scaling parameter

K = zeros(n, n);
K1 = zeros(n, n);
for i = 1:n
    for j = 1:n
        K(i,j) = exp(-(norm(X(i,:) - X(j,:))^2) / (2*sigma^2));
        K1(i,j) = exp(-(norm(X(i,:) - X(j,:))^2) / (2*sigma1^2));
    end
end

% Construct centered Gram matrices (K tilde)
In = (1/n) * ones(n, n);
Kt = K - In*K - K*In + In*K*In;
Kt1 = K1 - In*K1 - K1*In + In*K1*In;
k = 2;                      % Reduced dimension

% Eigenvalue decomposition (V: eigenvectors, L: eigenvalues)
[V, L] = eig(Kt);
[V1, L1] = eig(Kt1);

v = zeros(n, k);            % Initialization
v1 = zeros(n, k);
for j = 1:k                 % Extract first k components
    v(:,j) = V(:,j);
    v1(:,j) = V1(:,j);
end

% Projection for sigma = 0.4 (2D analysis)
y = v' * K;                 % Projected data in the k dimension
l = diag(L); l = l(1:k);
y = y ./ ((n-1)*l);
y = y';

% Projection for sigma1 = 0.8 (1D analysis)
y1 = v1' * K1;              % Projected data in the k dimension
l1 = diag(L1); l1 = l1(1:k);
y1 = y1 ./ ((n-1)*l1);
y1 = y1';

% Error calculation with Kernel PCA
errkPCA = norm(Kt - v*v'*Kt);
errrelkPCA = errkPCA / norm(Kt);
disp('PCA error='); disp(errrelkPCA);

errkPCA1 = norm(Kt1 - v1*v1'*Kt1);
errrelkPCA1 = errkPCA1 / norm(Kt1);
disp('PCA error='); disp(errrelkPCA1);

% Kernel PCA Plots
figure(4);
hold on;
gscatter(y1(1:M, 1), zero, labels_shuffled(1:M));
gscatter(y1(M+1:2*M, 1), zero, labels_shuffled(M+1:2*M));
gscatter(y1(2*M+1:end, 1), zero, labels_shuffled(2*M+1:end));
hold off;
xlabel('PC1');
title('First principal component after Kernel PCA');

figure(5);
hold on;
gscatter(y(1:M, 1), y(1:M, 2), labels_shuffled(1:M));
gscatter(y(M+1:2*M, 1), y(M+1:2*M, 2), labels_shuffled(M+1:2*M));
gscatter(y(2*M+1:end, 1), y(2*M+1:end, 2), labels_shuffled(2*M+1:end));
hold off;
xlabel('PC1');
ylabel('PC2');
title('First two principal component after Kernel PCA');

% --- Method 3: Isomap ---
M = points_per_cluster;
data = [X_shuffled Y_shuffled labels_shuffled];
data1 = [X_shuffled Y_shuffled];
distances = pdist(data1);
D1 = squareform(distances);
[Y, R, E] = IsomapFw(D1, 'k', 8);
