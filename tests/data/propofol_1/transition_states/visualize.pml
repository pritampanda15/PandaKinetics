# PandaKinetics Enhanced Visualization
# Ligand: CC(C)c1cccc(C(C)C)c1O
# Generated: 10 transition states

# Load all transition states
load state_000.pdb, state_000
load state_001.pdb, state_001
load state_002.pdb, state_002
load state_003.pdb, state_003
load state_004.pdb, state_004
load state_005.pdb, state_005
load state_006.pdb, state_006
load state_007.pdb, state_007
load state_008.pdb, state_008
load state_009.pdb, state_009

# Visualization settings
show sticks, all
set stick_radius, 0.15
spectrum b, blue_red, all

# Display settings
set ambient, 0.4
set specular, 1.0
orient all
zoom all, 3

save CC(C)c1cccc(C(C)C)c1O_visualization.pse

print 'PandaKinetics visualization loaded!'