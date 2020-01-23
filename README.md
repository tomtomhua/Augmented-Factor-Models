# The Reproduction of P02

This is a reproduction of P02 paper.<br>
Environment: Python 3.7 + Jupyter Lab (or Jupyter NoteBook)<br>
PreInstalled Packages: sklearn, pandas, pandas_datareader, numpy, matplotlib, datetime, scipy, joblib, multiprocessing, warnings<br>
CPUs: at least 5 cores for the computer<br>

## File usage

### Folders 
* data: all data used for the paper reproduction
* pic: the results for all pictures (save as png file)
* result: the results for all tables (save as csv file)
* temp: the data generated at intermediate steps for the convenience of use

### Documents
* function_to_use.py: All functions need to be used for simulations<br>
* MyHuberLoss.py: a modified version of sklearn.linear_model.HuberRegressor in order to satisfy the request of paper

* Stock_final.ipynb: An reproduction of Section5.<br>
* macro_final.ipynb: An reproduction of Section6 Table6.2.<br>
* macro_final_comparison.ipynb: An reproduction of Section6 Figure6.1<br>
* simulation7_i.ipynb: An reproduction of Table 7.i in Section7.

## Executions

For normal computers with Jupyter Notebook:<br>
run Stock_final.ipynb, macro_final.ipynb, simulation7_i.ipynb

For executions on HPC/CHTC or AWS/Google Cloud/...:<br>
change Stock_final.ipynb, macro_final.ipynb, simulation7_i.ipynb into Stock_final.py, macro_final.py, simulation7_i.py<br>
and run Stock_final.py, macro_final.py, simulation7_i.py

## Remarks

1. For those who does not have computers more than 5 CPUs,I recommend you to use HPC or CHTC resources at UWMadison, if so, you should first transform ".ipynb" files into ".py" files. The reason I use parallel is that I cannot finish even a single target before deadline without parallel running.
2. Due to the time limitation, I cannot run whole simulations (we have 6 cases, 5 cases with 200 replications each and the last one 1000 replications), since it takes **several hours** to run only **one case** **one simulation** with **5 Parallel CPUs**.
3. I write a reviesed version of Huber Loss function(see MyHuberLoss.py), and I cannot ensure its convergence under some cases. So I use "try ... except ..." structure to avoid the error, however, this will leads to an invalid data.
4. I just skip the reproducing of monthly stock data(Figure5.2 in paper), because I am not able to get the monthly data for all stocks continiously exists on S&P500 for many years.
5. I just skip the reproducing of Table 7.6, beacause it make no senses to check the FP rate without a number of replications, however it will take too huch time to do replications.
6. For Table 7.3, I just ignore the LS part, since the converence rate is comparatively slow and will takes a lot of time in calculations.
7. For all output data in Section 5,6, they might not make sense due to **the inaccuracy of data resource**.
8. For all output data in Section 7, they might not make sense due to **no replications**.


## Advice

1. The most important thing is to understand the code/algorithm and build up the frame work quickly. I mean that you may try but it might be a waste of time running the whole code to see whether some data gets wrong.
2. Use functional programming is a better choice, compared to code the procedure one by one.
3. I do not know how many days and nights I spend on this. So if you are struggled with exams afterwards, I highly recommend you just to reproduce some parts, or use simplifications on some functions.

## Good Luck, Code Fun
