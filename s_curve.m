% =========================================================================
% Title:       Nonlinear Dimensionality Reduction: 3D S-Curve Manifold
% Course:      Numerical Methods for Data Mining
% Author:      Chiara Giavalisco
% Description: Evaluates MDS, Kernel PCA (RBF kernel), and Isomap on a 
%              a 3D continuous manifold (S-Curve)
% =========================================================================

clear; close all; clc;

% --- Data Generation: S-Curve Manifold ---
N = 600;                        % Number of points
t = linspace(0, 2*pi, N);       % Parameter along the curve
z = t;
y = -sin(t) * 2;
x = rand(1, N) * 10;
X = [x; y; z]';                 % Matrix to store the S-curve points

% --- Visualization: Ground Truth ---
figure(1);
cmap = jet(N);
scatter3(x, y, z, 20, cmap);    % Colored scatter plot
title('S-Curve Manifold');
xlabel('X');
ylabel('Y');
zlabel('Z');
axis equal;

% --- Method 1: Multidimensional Scaling (MDS) ---
distances = pdist(X);
k = 3;                          % Target reduced dimension
D = squareform(distances);

% Double centering operation
I = eye(N);
O = ones(N);
H = I - ((1/N) * O);
B = -0.5 * H * D * H';

[U, L, W] = svd(B);
v = zeros(N, k);
s = zeros(k, k);
for j = 1:k
    v(:, j) = W(:, j);
    for i = 1:k
        s(i, j) = sqrt(L(i, j));
    end
end

z1 = s * v';
z1 = z1';                       % Projected data coordinates

% Error calculation with MDS
errMDS = norm((X'*X) - (z1'*z1));
errrelMDS = errMDS / norm(X'*X);
disp('MDS error='); disp(errrelMDS);

% MDS Plots
figure(2);
scatter(z1(:, 2), 0, 20, cmap);
xlabel('MD1');
title('Multidimensional Scaling onto 1D space');

figure(3);
scatter(z1(:, 2), z1(:, 3), 20, cmap);
xlabel('MD1');
ylabel('MD2');
title('Multidimensional Scaling onto 2D space');

% --- Method 2: Kernel PCA with Gaussian (RBF) Kernel ---
K = zeros(N);                   % Initialization
sigma = 2.2; 
for i = 1:N
    for j = 1:N
        K(i, j) = exp(-(norm(data(i, :) - data(j, :))^2) / (2 * (sigma)^2));
    end
end

In = (1/N) * ones(N);
Kt = K - In*K - K*In + In*K*In; % Centered Gram matrix (K tilde)

% Eigenvalue decomposition (V: eigenvectors, L: eigenvalues)
[V, L] = eig(Kt); 

% Normalization of V
norm_eigVector = sqrt(sum(V.^2));
V = V ./ repmat(norm_eigVector, size(V, 1), 1);

% Dimensionality reduction
V = V(:, 1:k);
y = V' * K;                     % Projected data in the k dimension
Diag = diag(L); 
[l, ~] = sort(Diag, 'descend'); % Eigenvalues in descending order
l = l(1:k);
y = y ./ ((N-1) .* l);
% y = y';

% Error calculation with Kernel PCA
errkPCA = norm(Kt - V*V'*Kt); 
errrelkPCA = errkPCA / norm(Kt); 
disp('PCA error='); disp(errrelkPCA);

% Kernel PCA Plots
figure(4);
hold on;
scatter(y(2, 1:N), 0, 20, cmap);
hold off;
xlabel('PCA1');
title('First principal component after Kernel PCA');

figure(5);
hold on;
scatter(y(2, 1:N), y(3, 1:N), 20, cmap);
hold off;
xlabel('PCA1');
ylabel('PCA2');
title('First two principal component after Kernel PCA');

% --- Method 3: Isomap ---
[Y, R, E] = IsomapScurve(D, 'k', 8);
