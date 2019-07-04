# GeneticMLP
Implementation of a Multilayer Perceptron in Keras using a Genetic Algorithm (PonyGE2 Grammatical Evolution)

The code of interest (my implementation) is the following:

* src/fitness/MLP.py: fitness function containing the GE fitness evaluation function.
* grammars/mlpGrammar.pybnf: Grammar describing the algorithm genotype. Dynamic Keras code is generated.
