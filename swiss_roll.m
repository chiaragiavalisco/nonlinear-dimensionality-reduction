% =========================================================================
% Title:       Nonlinear Dimensionality Reduction: 3D Swiss Roll
% Course:      Numerical Methods for Data Mining
% Author:      Chiara Giavalisco
% Description: Evaluates MDS, Kernel PCA (RBF kernel), and Isomap on a 
%              canonical 3D unrolling manifold task (Swiss Roll).
% =========================================================================

clear; clc; close all;

% --- Data Generation: Swiss Roll ---
N = 2048;                       % Number of sampled points
t = rand(1, N);
t = sort(4 * pi * sqrt(t))';    % Parameter parameterizing spiral windings
z = 8 * pi * rand(N, 1);
x = (t + 0.1) .* cos(t);
y = (t + 0.1) .* sin(t);
data = [x, y, z];

% Plot 3D Manifold
figure(1);
cmap = jet(N);                  % Color map mapped by spiral progression
scatter3(x, y, z, 20, cmap);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Swiss Roll');
view(-30, 20);

% --- Method 1: Multidimensional Scaling (MDS) ---
distances = pdist(data);
k = 3;                          % Extracted dimensions
D = squareform(distances);

% Double centering
I = eye(N);
O = ones(N);
H = I - ((1/N)*O);
B = -0.5 * H * D * H';

[~, L, W] = svd(B);
v = zeros(N, k);
s = zeros(k, k);
for j = 1:k
    v(:, j) = W(:, j);
    for i = 1:k
        s(i, j) = sqrt(L(i, j));
    end
end

z1 = (s * v')';                 % Projected coordinates

% Error calculation
errMDS = norm((data'*data) - (z1'*z1));
errrelMDS = errMDS / norm(data'*data);
disp(['MDS relative error: ', num2str(errrelMDS)]);

figure(2);
scatter(z1(:, 2), 0, 20, cmap);
xlabel('MD1');
title('1D Multidimensional Scaling');

figure(3);
scatter(z1(:, 2), z1(:, 3), 20, cmap);
xlabel('MD1'); ylabel('MD2');
title('2D Multidimensional Scaling');

% --- Method 2: Kernel PCA (Gaussian / RBF Kernel) ---
sigma = 15.8;
K = zeros(N);
for i = 1:N
    for j = 1:N
        K(i, j) = exp(-(norm(data(i, :) - data(j, :))^2) / (2 * (sigma^2)));
    end
end

In = (1/N) * ones(N);
Kt = K - In*K - K*In + In*K*In;

[V, L_eig] = eig(Kt);

% Normalization of eigenvectors
norm_eigVector = sqrt(sum(V.^2));
V = V ./ repmat(norm_eigVector, size(V, 1), 1);

% Dimensionality reduction
V = V(:, 1:k);
y = V' * K;
Diag = diag(L_eig);
[l, ~] = sort(Diag, 'descend');
l = l(1:k);[cite: 1]
y = y ./ ((N - 1) .* l);

% Error calculation
errkPCA = norm(Kt - V * V' * Kt);
errrelkPCA = errkPCA / norm(Kt);
disp(['KPCA relative error: ', num2str(errrelkPCA)]);

figure(4);
scatter(y(2, :), 0, 20, cmap);
xlabel('PC1');
title('First Principal Component after Kernel PCA');

figure(5);
scatter(y(2, :), y(3, :), 20, cmap);
xlabel('PC1'); ylabel('PC2');
title('First Two Principal Components after Kernel PCA');

% --- Method 3: Isomap ---
[Y, R, E] = IsomapSwissRoll(D, 'k', 8);
