# GeneticMLP

[WORK IN PROGRESS]

Implementation of a simple Multilayer Perceptron in Keras using PonyGE2 Genetic Algorithm (https://github.com/PonyGE/PonyGE2).

The code of interest (my implementation) is the following:

* src/fitness/MLP.py: fitness function containing the GE fitness evaluation function.
* grammars/mlpGrammar.pybnf: Grammar (in python Backus-Naur Form) describing the algorithm genotype. Dynamic Keras code is generated.

TODO:
+ Improve grammar adding ActivationFunction selection, Optimization selector,...
+ Add computation of score (classification accuracy in keras)
+ Test valid results/ files creation 
+ Code final model retrieval/evaluation functions
