# VGA Experimental Dataset

**Pedestrian Navigation Experiments - Real Human Data**

---

## Overview

This dataset contains real human pedestrian navigation experiments conducted in controlled environments. The data includes start/goal positions, obstacle configurations, desired speeds, and experiment identifiers across multiple scenarios.

**Source:** [Pedestrian-Experimental-Data](https://github.com/kanika201293/Pedestrian-Experimental-Data)

**Total Experiments:** 941 trials across 7 scenarios

---

## Experimental Scenarios

There are multiple experiments: SOSP, Head-On, Parallel-Ped, MOSP (Case A, B, C, and D). The experimental data is provided in the following files:

---

### SOSP (Single Obstacle Single Pedestrian)  
`1.`'SOSP_initialFinalPos_feed.txt' (6 columns) - pedInitialPos_x, pedInitialPos_y, pedFinalPos_x, pedFinalPos_y, pedDesiredSpeed, ExpNo.  
`2.`'SOSP_obstPos_feed.txt' (3 columns) - obstPos_x, obstPos_y, obstNo.  
`3.` SOSP_maxTime = 6.833 s
      
Head-On:  
`1.`'Head_On_initialFinalPos_feed.txt' (11 columns) - ped1InitialPos_x, ped1InitialPos_y, ped1FinalPos_x, ped1FinalPos_y, ped1DesiredSpeed, ped2InitialPos_x, ped2InitialPos_y, ped2FinalPos_x, ped2FinalPos_y, ped2DesiredSpeed, ExpNo.  
`2.` Head_On_maxTime = 8.033 s

Parallel-Ped:  
`1.`'Parallel_Ped_initialFinalPos_feed.txt' (11 columns) - ped1InitialPos_x, ped1InitialPos_y, ped1FinalPos_x, ped1FinalPos_y, ped1DesiredSpeed, ped2InitialPos_x, ped2InitialPos_y, ped2FinalPos_x, ped2FinalPos_y, ped2DesiredSpeed, ExpNo.  
`2.` Parallel_Ped_maxTime = 12.033 s

MOSP (Case A):  
`1.`'MOSP_CaseA_initialFinalPos_feed.txt' (7 columns) - pedInitialPos_x, pedInitialPos_y, pedFinalPos_x, pedFinalPos_y, pedDesiredSpeed, ExpNo., idxNo. (no use)  
`2.`'MOSP_CaseA_obstPos_feed.txt' (3 columns) - obstPos_x, obstPos_y, obstNo.  
`3.` MOSP_CaseA_maxTime = 10.866 s
               
MOSP (Case B):  
`1.`'MOSP_CaseB_initialFinalPos_feed.txt' (7 columns) - pedInitialPos_x, pedInitialPos_y, pedFinalPos_x, pedFinalPos_y, pedDesiredSpeed, ExpNo., idxNo. (no use)  
`2.`'MOSP_CaseB_obstPos_feed.txt' (3 columns) - obstPos_x, obstPos_y, obstNo.  
`3.` MOSP_CaseB_maxTime = 11.266 s
               
MOSP (Case C):  
`1.`'MOSP_CaseC_initialFinalPos_feed.txt' (7 columns) - pedInitialPos_x, pedInitialPos_y, pedFinalPos_x, pedFinalPos_y, pedDesiredSpeed, ExpNo., idxNo. (no use)  
`2.`'MOSP_CaseC_obstPos_feed.txt' (3 columns) - obstPos_x, obstPos_y, obstNo.  
`3.` MOSP_CaseC_maxTime = 11.466 s
               
MOSP (Case D):  
`1.`'MOSP_CaseD_initialFinalPos_feed.txt' (7 columns) - pedInitialPos_x, pedInitialPos_y, pedFinalPos_x, pedFinalPos_y, pedDesiredSpeed, ExpNo., idxNo. (no use)  
`2.`'MOSP_CaseD_obstPos_feed.txt' (3 columns) - obstPos_x, obstPos_y, obstNo.  
`3.` MOSP_CaseD_maxTime = 12.833 s
