# PandaKinetics Transition States Visualization
# Ibuprofen binding pathway

# Load all states
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

# Color by binding energy
# Blue = favorable, Red = unfavorable
color blue, state_000
color blue, state_001
color blue, state_002
color blue, state_003
color blue, state_004
color blue, state_005
color green, state_006
color green, state_007
color green, state_008
color green, state_009

# Display settings
show sticks, all
set stick_radius, 0.15

# Create energy labels
label state_000 and name C1, "-12.9"
label state_001 and name C1, "-11.0"
label state_002 and name C1, "-10.5"
label state_003 and name C1, "-10.5"
label state_004 and name C1, "-10.0"
label state_005 and name C1, "-10.0"
label state_006 and name C1, "-10.0"
label state_007 and name C1, "-9.8"
label state_008 and name C1, "-9.7"
label state_009 and name C1, "-9.6"

# Set view
orient all
zoom all, 5

# Save session
save ibuprofen_pathway.pse