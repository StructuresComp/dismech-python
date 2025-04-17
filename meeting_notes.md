# Meeting Notes

## 2-27-25
- [x] vectorize energy computations
- [x] abstract away inertial force (input: old DOF, velocity, acceleration)
- [x] remove external force in different file than TimeStepper
- [x] check about energy initialization
- [x] boundary condition est functions (have time as input ? -- ABAQUS disp())
- [x] different types of integration schemes (Newmark Beta??)
- [x] pardiso ? (pip install)
- [x] beforeTimestep fn ?

## 4-16-25

### Springs and ElasticEnergy
- [ ] standardize spring convention
- [ ] natural and incremental strain (spring wise)
- [ ] allow spring changes in before_step to propogate to energies

### External forces
- [ ] create external force abstract class
- [ ] streamline aerodynamic calculations (for sparse)

### TimeStepper
- [ ] Dictionary access to energy objects
- [ ] Line search

### Visualization
- [ ] Make it both python and ipynb friendly