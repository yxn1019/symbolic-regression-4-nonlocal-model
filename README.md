# Manuscript Title
Discovering Memory Functions and Governing Equations for Non-Fickian Transport in Aquatic Systems via Nonlocal Theory and Symbolic Regression

## Manuscript's cases
  'submission_sr.pdf' is the manuscript of our study, including 6 cases, where
	case 1-3 for mass transfer,
	case 4-6 for heterogeneous conductivity.

**'example_stm.py' is a brief example of discovery process.**


## To implement model discovery task: 

Step 1. Run 'case1_ctrw_syn.py'/'case2_ctrw_cr.py'/'case3_ctrw_river.py' for mass transfer scenario, or
    run 'case4-6_stm.py' for heterogeneous conductivity scenario.
	We obtain the memory kernel expression. The expression used for our cases is stored in folder 'outputs', the folder name can be checked in 'xx.py' mentioned above.

Step 2. Embed the kernel expression into 'main_ctrw_sr.m'/'main_stm.str.m'  to get the breakthrough-curve/snapshot, where 'nil.m' is the program for numerical inverse Laplace transform, 'sr_FADE.m' is the mathematical formula of our ctrw model.

> All the datasets have been smoothed by surrogate models and stored in folder 'model_save', if new scenarios is considered, you could use 'NN_ode.py' to smooth the data.
 
