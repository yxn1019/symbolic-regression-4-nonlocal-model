# Manuscript Title
Discovering Memory Functions and Governing Equations for Non-Fickian Transport in Aquatic Systems via Nonlocal Theory and Symbolic Regression

## File description
Folder 'dataset' includes the datasets in our manuscript;
Folder 'model_save' provides the reconstructed data implemented by DNN, and 'NN_ode.py' is the program traning the data by DNN.
Folder 'outputs' saves the results and model's information of symbolic regression;
'submission_sr.pdf' is the manuscript of our study, including 6 cases, where
	case 1-3 for mass transfer, the corresponding code is 'casex_CTRW_xx.py' 
	case 4-6 for heterogeneous conductivity, the corresponding code is 'case4-6_stm.py';
'data_combine_stm_btc.m'/'data_combine_stm_snapshot.m' implement the data transformation for streamtube model from btc/snapshots to the pdf of velocity via variable transformation.
'example_stm/ctrw.py' is a detailed example of discovery process.
'main_ctrw_sr.m'/'main_stm_sr.m' is the numerical solution for CTRW/streamtube models, where 'sr_FADE.m' is the semi-analytical solution of CTRW models with custom memory function.

## Brief start
Directly run 'example_stm/ctrw.py' for streamtube model (stm) and Continous Time Random Walk (ctrw) respectively, and obtain the formula of memory function.


## To implement model discovery task

### The cases in manuscript:
Step 1. Run 'case1_ctrw_syn.py'/'case2_ctrw_cr.py'/'case3_ctrw_river.py' for mass transfer scenario, or
    run 'case4-6_stm.py' for heterogeneous conductivity scenario.
	Then we obtain the memory kernel expression (The expression formula and the corresponding metadata has been save in folder 'outputs').

Step 2. Embed the kernel formula into 'main_ctrw_sr.m'/'main_stm_sr.m' to get the breakthrough-curve/snapshot, where 'nil.m' is the program for numerical inverse Laplace transform, 'sr_FADE.m' is the mathematical formula of our ctrw model.

> All the datasets have been smoothed by surrogate models and stored in folder 'model_save', if new scenarios is considered, you could use 'NN_ode.py' to smooth the data.

### The custom task:
*CTRW*
Step 1. Put the custom dataset into folder 'dataset'.<br>
Step 2. Run 'NN_ode.py' to obtain the high-quality data, and the results are saved in folder 'model_save', pay attention to file naming.<br>
Step 3. Replace the old filename in 'example_ctrw_cr.py'(single detection point)/'example_ctrw_river.py'(multi-detection point) with the new one, and run the code. Then we obtain the memory kernel expression.<br>
Step 4. Import the formula into 'main_ctrw_sr.m', and run the code to get the breakthrough curve.<br>
*Streamtube model*<br>
*CTRW*<br>
Step 1. Put the custom dataset into folder 'dataset'.<br>
Step 2. Run 'data_combine_stm_btc.m'/'data_combine_stm_snapshot.m' to obtain the pdf of velocity from breakthrough curve/snapshot observations, and the pdf data should be saved in folder 'dataset' with custom name.<br>
Step 3. Import the pdf data (with custom name) into 'example_stm.py', and run the code. Then we obtain the memory kernel expression.<br>
Step 4. Import the formula into 'main_stm_sr.m', and run it to get the breakthrough curve/snapshot.<br>
 
