# OMML-HW2

This repository contains the code for the second homework of the *Optimization Methods for Machine Learning* course (1041415), Sapienza University. The group was composed by Nina Kaploukhaya, Mehrdad Hassanzadeh and me.

**TL;DR**: The task involved building different optimization methods for training a non-linear SVM to solve a classification problem.

## Data

The dataset chosen is MNIST, for the binary classification problem of Question 1, 2 and 3 only digit 1 and digit 5 are considered. For Question 4, images containing the digit 7 are also included.

## Models

The model trained is *non-linear SVM* with *polynomial kernel*, the hyperparameters $C$ and $\gamma$ should be chosen with an heuristic procedure, whereas the parameters $\alpha$ and $\beta$ should be found with different optimization procedures.

For Question 1, the task is to find the solution of the *SVM dual quadratic problem*, and the chosen optimization routine is *cvxopt.solvers.qp*.

For Question 2, the task is to implement a *Decomposition Method* for the dual quadratic problem with any even value $q \geq 4$. Also here *cvxopt.solvers.qp* is the routine chosen for the optimization.

For Question 3, the dimension of the subproblem is fixed to $q = 2$, hence the decomposition method adopted is a *Sequential Minimal Optimization* algorithm whose solution can be found analytically (see `Algorithm 5.A.1.png` in `stuff` folder)!

For Question 4, a *one-against-all* SVM model was trained for classifying observation from each one of the three classes (ie., 1, 5 and 7). 

## Results 
Everything is explained in `Report.pdf`
