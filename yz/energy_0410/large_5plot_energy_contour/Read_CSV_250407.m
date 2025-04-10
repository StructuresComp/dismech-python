
clear; clc;

csvFile = 'final_nodal_coordinates.csv';

% Read the CSV file as a table
T = readtable(csvFile);

% The columns are:
% 1) Instance (categorical or string)
% 2) NodeLabel
% 3) U1
% 4) U2
% 5) U3
% 6) X
% 7) Y
% 8) Z
% 9) X_final
% 10) Y_final
% 11) Z_final

kirigamiTable  = T(strcmp(T.Instance, 'KIRIGAMI-LAYER'), :);
substrateTable = T(strcmp(T.Instance, 'SUBSTRATE-LAYER'), :);

kirigamiArray = [ ...
    kirigamiTable.NodeLabel, ...
    kirigamiTable.U1, ...
    kirigamiTable.U2, ...
    kirigamiTable.U3, ...
    kirigamiTable.X, ...
    kirigamiTable.Y, ...
    kirigamiTable.Z, ...
    kirigamiTable.X_final, ...
    kirigamiTable.Y_final, ...
    kirigamiTable.Z_final ...
];

substrateArray = [ ...
    substrateTable.NodeLabel, ...
    substrateTable.U1, ...
    substrateTable.U2, ...
    substrateTable.U3, ...
    substrateTable.X, ...
    substrateTable.Y, ...
    substrateTable.Z, ...
    substrateTable.X_final, ...
    substrateTable.Y_final, ...
    substrateTable.Z_final ...
];

% Optionally, save the arrays to a MAT file
savePath = 'Kirigami_Substrate_Data.mat';
save(savePath, 'kirigamiArray', 'substrateArray');


figure(1)
scatter3(kirigamiArray(:,5), kirigamiArray(:,6), kirigamiArray(:,7), 'b', 'filled');
hold on
scatter3(kirigamiArray(:,8), kirigamiArray(:,9), kirigamiArray(:,10), 'ro' ); % 'filled'
xlabel('X');
ylabel('Y');
zlabel('Z');
legend('reference','current')
title('Kirigami');
grid on;
axis equal;


figure(2)
scatter3(kirigamiArray(:,5), kirigamiArray(:,6), kirigamiArray(:,7), 'b', 'filled');
hold on
scatter3(substrateArray(:,5), substrateArray(:,6), substrateArray(:,7), 'ro');
xlabel('X');
ylabel('Y');
zlabel('Z');
legend('kirigami','substrate')
title('Original Coordinates');
grid on;
% axis equal;


figure(3)
scatter3(kirigamiArray(:,8), kirigamiArray(:,9), kirigamiArray(:,10), 'b', 'filled');
hold on
scatter3(substrateArray(:,8), substrateArray(:,9), substrateArray(:,10), 'ro' );
xlabel('X');
ylabel('Y');
zlabel('Z');
legend('reference','current')
title('Substrate');
grid on;
axis equal;



%% reference coordinates

fun_input(kirigamiArray, 'connectivity_matrix.csv', 'input_ref_kirigami_dismech.dat', 'KIRIGAMI-LAYER', 'reference');
fun_input(kirigamiArray, 'connectivity_matrix.csv', 'input_cur_kirigami_dismech.dat', 'SUBSTRATE-LAYER', 'current');




% % Sort nodes by NodeLabel in ascending order.
% [sortedLabels, idx] = sort(kirigamiArray(:,1));
% sortedCoords = kirigamiArray(idx, 5:7);  % Original coordinates: X, Y, Z
% 
% conn_csvFile = 'connectivity_matrix.csv';
% connT = readtable(conn_csvFile, 'ReadVariableNames', false, 'Delimiter', ',');
% % Rename variables
% connT.Properties.VariableNames = {'Instance','ElementLabel','Node1','Node2','Node3'};
% 
% % Select connectivity for 'KIRIGAMI-LAYER'
% connTable = connT(strcmp(connT.Instance, 'KIRIGAMI-LAYER'), :);
% 
% outFile = 'input_ref_dismech.dat';
% fid = fopen(outFile, 'w');
% if fid == -1
%     error('Cannot open file %s for writing.', outFile);
% end
% 
% % Write the *Nodes section.
% fprintf(fid, '*Nodes\n');
% for i = 1:size(sortedCoords, 1)
%     fprintf(fid, '%f, %f, %f\n', sortedCoords(i,1), sortedCoords(i,2), sortedCoords(i,3));
% end
% % fprintf(fid, '\n');
% 
% % Write the *Triangles (connectivity) section.
% fprintf(fid, '*Triangles\n');
% for i = 1:height(connTable)
%     % Write the connectivity as: Node1, Node2, Node3
%     fprintf(fid, '%d, %d, %d\n', connTable.Node1(i), connTable.Node2(i), connTable.Node3(i));
% end
% 
% fclose(fid);
% disp(['Formatted input file saved as ', outFile]);


%% current coordinates
