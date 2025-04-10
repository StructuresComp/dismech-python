

%%
nodeFile = 'input_cur_kirigami_dismech.dat';
fid = fopen(nodeFile, 'r');
if fid < 0
    error('Could not open file: %s', nodeFile);
end

% Read until we find '*Nodes'
line = fgetl(fid);
while ischar(line) && ~contains(line, '*Nodes')
    line = fgetl(fid);
end

% Now read node lines until we hit '*Triangles' or EOF
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

%%
% Format: EdgeID, N1, N2
connFile = 'connectivity.csv';
connData = readmatrix(connFile);

% NOTE: readmatrix will put them in columns 1-based in MATLAB indexing:
edgeID  = connData(:, 1);
node1   = connData(:, 2);
node2   = connData(:, 3);

edges = [node1, node2];

%%
energyFile = 'cur_kirigami_dismech_energy.csv';
energyData = readmatrix(energyFile);
% Index, EdgeID, HingeStrain, HingeEnergy, StretchStrain, StretchEnergy,

energy_edgeID = energyData(:, 2);
stretchEnergyAll = energyData(:, 6);

numEdges = size(edges, 1);
stretchEnergy = zeros(numEdges, 1);

% The EdgeID from Python is 0-based, while MATLAB is 1-based
for i = 1:length(energy_edgeID)
    thisEID = energy_edgeID(i);

    if thisEID >= 0 && thisEID < numEdges
        val = stretchEnergyAll(i);
        % clamp any huge negative placeholders:
%         if val < -1e9
%             val = 0;  % or skip
%         end
        stretchEnergy(thisEID+1) = val;
    end
end

%%
figure('Name','Truss 3D Plot','Color','white');
hold on; axis equal; grid on;

% colormap 'jet', 'parula', 'turbo'.
colormap('parula');

cmin = min(stretchEnergy);
cmax = max(stretchEnergy);
caxis([cmin cmax]);  % Set color axis for the colorbar
colorbar;            

for i = 1:numEdges
    % Node indices for the i-th edge:
    n1 = edges(i,1) + 1;  % shift 0-based to 1-based
    n2 = edges(i,2) + 1;

    % Coordinates for the two endpoints
    x = [nodes(n1,1), nodes(n2,1)];
    y = [nodes(n1,2), nodes(n2,2)];
    z = [nodes(n1,3), nodes(n2,3)];

    % Current edge’s stretch energy
    val = stretchEnergy(i);

    % Convert stretch energy by a linear mapping from [cmin, cmax] into [1, size_of_colormap]
    cMap = colormap;
    nColors = size(cMap,1);

    if cmax == cmin
        % Avoid divide-by-zero if all energies are the same
        colorIdx = 1;
    else
        colorIdx = round( (val - cmin)/(cmax - cmin)*(nColors-1) ) + 1;
    end

    % Bound the index
    colorIdx = max(1, min(nColors, colorIdx));

    % Get the actual RGB color
    rgbColor = cMap(colorIdx, :);

    plot3(x, y, z, 'LineWidth', 2, 'Color', rgbColor);
end

xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Truss with Links Colored by Stretch Energy');
hold off;

