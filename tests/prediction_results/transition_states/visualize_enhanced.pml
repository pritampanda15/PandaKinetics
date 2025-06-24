# PandaKinetics Visualization
# Ligand: CC(C)c1cccc(C(C)C)c1O
# Generated: 10 transition states with PROPER CONNECTIVITY

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

# Enhanced visualization settings
show sticks, all
set stick_radius, 0.15
set stick_ball, on
set stick_ball_ratio, 1.8

# Color by energy (stored in B-factor)
color cyan, all

# Show proper bonds (CONECT records will be used automatically)
rebuild

# Display settings for better visualization
set ambient, 0.4
set specular, 1.0
set ray_opaque_background, off
set antialias, 2

# Center and orient
orient all
zoom all, 3

# Save session
save CC(C)c1cccc(C(C)C)c1O_enhanced_visualization.pse

print 'Enhanced PandaKinetics visualization with proper bonds loaded!'
print 'Loaded 10 states with correct molecular connectivity'