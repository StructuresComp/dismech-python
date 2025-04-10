

nodeFile_ref='input_ref_kirigami_dismech.dat';
current_config_energy_file='cur_kirigami_dismech_energy.csv';
energyData = readmatrix(current_config_energy_file);


%% Plot

ContourVec = energyData(:,5);
titleString='stretch strain';
fun_shell_energy_v3('input_cur_kirigami_dismech.dat', 'connectivity.csv', current_config_energy_file, ContourVec, titleString, nodeFile_ref);

ContourVec = energyData(:,6);
titleString='stretch energy';
fun_shell_energy_v3('input_cur_kirigami_dismech.dat', 'connectivity.csv', current_config_energy_file, ContourVec, titleString, nodeFile_ref);


ContourVec = energyData(:,3);
titleString='hinge strain';
fun_shell_energy_v3('input_cur_kirigami_dismech.dat', 'connectivity.csv', current_config_energy_file, ContourVec, titleString, nodeFile_ref);

ContourVec = energyData(:,4);
titleString='hinge energy';
fun_shell_energy_v3('input_cur_kirigami_dismech.dat', 'connectivity.csv', current_config_energy_file, ContourVec, titleString, nodeFile_ref);


%% statics

stretchEnergyAll = energyData(:,5);
validStretchEnergy = stretchEnergyAll(stretchEnergyAll > -1e9);
figure
histogram(validStretchEnergy, 'NumBins', 500, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('stretch strain');
ylabel('Frequency');
grid on;

stretchEnergyAll = energyData(:,6);
validStretchEnergy = stretchEnergyAll(stretchEnergyAll > -1e9);
figure('Color', 'white');
histogram(validStretchEnergy, 'NumBins', 500, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('stretch Energy');
ylabel('Frequency');
grid on;


stretchEnergyAll = energyData(:,3);
validStretchEnergy = stretchEnergyAll(stretchEnergyAll > -1e9);
figure
histogram(validStretchEnergy, 'NumBins', 500, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('hinge strain');
ylabel('Frequency');
grid on;

stretchEnergyAll = energyData(:,4);
validStretchEnergy = stretchEnergyAll(stretchEnergyAll > -1e9);
figure
histogram(validStretchEnergy, 'NumBins', 500, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('hinge energy');
ylabel('Frequency');
grid on;




x = energyData(:,5);
x = x(x > -1e9);
y = energyData(:,6);
y = y(y > -1e9);
figure
plot(x,y,'o');
xlabel('stretch strain');
ylabel('stretch energy');
grid on;



x = energyData(:,3);
x = x(x > -1e9);
y = energyData(:,4);
y = y(y > -1e9);
figure
plot(x,y,'o');
xlabel('hinge strain');
ylabel('hinge energy');
grid on;




