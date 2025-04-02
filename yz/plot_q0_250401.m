clear; clc;

% Load and reshape the node coordinates
load('base_robot_q0.txt')                 % base_robot_q0 is 81x1
points = reshape(base_robot_q0, 3, [])';  % becomes 27x3

% Load the connectivity
T    = readtable('connectivity.csv');
N1   = T.N1;   % zero-based index
N2   = T.N2; 
EdgeID = T.EdgeID;

figure;
hold on;

plot3(points(:,1), points(:,2), points(:,3), 'o', ...
      'MarkerFaceColor','b','MarkerEdgeColor','k','MarkerSize',8);

for i = 1:size(points,1)
    text(points(i,1), points(i,2), points(i,3), sprintf(' %d', i-1), ...
         'Color','k','FontSize',15, 'FontWeight', 'bold');
end

% Draw edges
for i = 1:length(N1)
    % Convert zero-based indices to MATLAB's one-based indexing
    p1 = points(N1(i)+1, :);
    p2 = points(N2(i)+1, :);
    
    plot3([p1(1), p2(1)], [p1(2), p2(2)], [p1(3), p2(3)], 'Color', [0 0.5 0], 'LineWidth', 1.5);
    
    midPoint = (p1 + p2) / 2;
    
    text(midPoint(1), midPoint(2), midPoint(3), sprintf(' %d', EdgeID(i)), ...
         'Color','r','FontSize',15, 'FontWeight', 'bold');
end

xlabel('X');
ylabel('Y');
zlabel('Z');
title('Base Robot Nodes and Edges with IDs');
grid on;
axis equal; 
hold off;
