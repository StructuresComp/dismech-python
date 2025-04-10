function fun_shell_energy_v3(nodeFile, connFile, energyFile, stretchEnergyColumn, titleString, nodeFile_ref)
% PLOT_TRUSS_WITH_ENERGY plots a 3D truss with links colored by stretch energy,
% and plots a black line for placeholder edges (val < -1e9).
%
% excludes the placeholder values from min/max color scaling.

%% Read node coordinates
fid = fopen(nodeFile, 'r');
if fid < 0, error('Could not open file: %s', nodeFile); end
line = fgetl(fid);
while ischar(line) && ~contains(line, '*Nodes')
    line = fgetl(fid);
end
nodes = [];
line = fgetl(fid);
while ischar(line) && ~contains(line, '*Triangles')
    nums = sscanf(line, '%f, %f, %f');
    if numel(nums) == 3
        nodes = [nodes; nums(:)'];
    end
    line = fgetl(fid);
end
fclose(fid);

%% Read connectivity
connData = readmatrix(connFile); % Format: EdgeID, N1, N2
edgeID  = connData(:, 1);
node1   = connData(:, 2);
node2   = connData(:, 3);
edges = [node1, node2];

%% Read energy data and map stretch energy to edges
energyData = readmatrix(energyFile);
energy_edgeID = energyData(:, 2);
stretchEnergyAll = stretchEnergyColumn;
numEdges = size(edges, 1);
stretchEnergy = zeros(numEdges, 1);
for i = 1:length(energy_edgeID)
    thisEID = energy_edgeID(i);
    if thisEID >= 0 && thisEID < numEdges
        val = stretchEnergyAll(i);
        stretchEnergy(thisEID+1) = val;
    end
end

%% Exclude placeholder values (< -1e9) from color scaling
validValues = stretchEnergy(stretchEnergy > -1e9);
cmin = min(validValues);
cmax = max(validValues);

%% Plot 3D truss colored by stretch energy
figure('Name','Truss 3D Plot','Color','white');
hold on; axis equal; grid on;
colormap('parula');
caxis([cmin cmax]);
colorbar;

for i = 1:numEdges
    n1 = edges(i, 1) + 1; 
    n2 = edges(i, 2) + 1;
    x = [nodes(n1, 1), nodes(n2, 1)];
    y = [nodes(n1, 2), nodes(n2, 2)];
    z = [nodes(n1, 3), nodes(n2, 3)];
    val = stretchEnergy(i);
    
    % Plot placeholder energy edges in black
    if val < -1e9
        rgbColor = [0, 0, 0];
    else
        cMap = colormap;
        nColors = size(cMap, 1);
        if cmax == cmin
            colorIdx = 1;
        else
            colorIdx = round((val - cmin)/(cmax - cmin) * (nColors - 1)) + 1;
        end
        colorIdx = max(1, min(nColors, colorIdx));
        rgbColor = cMap(colorIdx, :);
    end
    
    plot3(x, y, z, 'LineWidth', 3, 'Color', rgbColor);
end


%% Plot ref config.
fid = fopen(nodeFile_ref, 'r');
if fid < 0, error('Could not open file: %s', nodeFile_ref); end
line = fgetl(fid);
while ischar(line) && ~contains(line, '*Nodes')
    line = fgetl(fid);
end
nodes_ref = [];
line = fgetl(fid);
while ischar(line) && ~contains(line, '*Triangles')
    nums = sscanf(line, '%f, %f, %f');
    if numel(nums) == 3
        nodes_ref = [nodes_ref; nums(:)'];
    end
    line = fgetl(fid);
end
fclose(fid);

rgbColor = [0.8, 0.8, 0.8];
for i = 1:numEdges
    n1 = edges(i, 1) + 1; 
    n2 = edges(i, 2) + 1;
    x = [nodes_ref(n1, 1), nodes_ref(n2, 1)];
    y = [nodes_ref(n1, 2), nodes_ref(n2, 2)];
    z = [nodes_ref(n1, 3), nodes_ref(n2, 3)];    
    plot3(x, y, z, 'LineWidth', 1.5, 'Color', rgbColor);
end



xlabel('X'); ylabel('Y'); zlabel('Z');
title(titleString);
hold off;

