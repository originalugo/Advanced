<b>Task <br /></b>
Task is to implement an evolutionary algorithm and apply it to the 
problem shown below. You will do a variety of experiments to help find out what parameters for the 
algorithm are best for this problem. Implementation of the algorithm can be in the programming 
language of your choice. In the following sections, details of the problem are provided, then basic details 
of the algorithm, and finally a description of the experiments you should carry out. The final section 
indicates what should be in your report to be handed in. 

<strong>The Problem</strong><br/>
Working for a bank, you have been asked to develop an evolutionary algorithm based system which will 
find the largest amount of money that can be packed into a security van. The money is separated into 
100 bags of different denominations and the weight and value of the money of each bag is shown on 
the outside of the bag, e.g., 
Bag 1 Weight = 9.4Kg, Value = £57
Bag 2 Weight = 7.4Kg, Value = £94
. 
. 
. 
Bag N Weight = iKg, Value = £x, </br>

The security van can carry only a limited weight, so your system must try and decide which bags to put 
on the van, and which ones to leave behind. The best solution will be the one which packs the most 
money (in terms of value) into the van without overloading it. </br>

• Your system should read in the 100 bag values from the file “BankProblem.txt” which is 
provided along with this document. </br>
• The file contains the weight limit for the security van and the values and weights for each 
bag of money. Weights are all in kilos and the values are all in pounds sterling. </br>
• You must decide how to represent this problem to the evolutionary algorithm, you must 
also decide what the fitness function should be. </br>

<strong> The Evolutionary Algorithm </strong><br/>
The evolutionary algorithm should be implemented as follows: 
1. Generate an initial population of p randomly generated solutions (where p is a reasonable 
population size discussed in lectures and in the module reading), and evaluate the fitness of 
everything in the population. 
2. Use the binary tournament selection twice (with replacement) to select two parents a and b. 
3. Run crossover on these parents to give 2 children, c and d. 
4. Run mutation on c and d to give two new solutions e and f. Evaluate the fitness of e and f. 
5. Run weakest replacement, first using e, then f. 
6. If a termination criterion has been reached, then stop. Otherwise return to step 2. 

<strong>Termination Criterion:</strong> Will simply be having reached a maximum number of fitness evaluations which 
is 10,000 (see Implementation and Experimentation below) </br>

<strong>Binary Tournament Selection:</strong> Randomly choose a chromosome from the population; call it a. Randomly 
choose another chromosome from the population; call this b. The fittest of these two (breaking ties 
randomly) becomes the selected parent. </br>

<strong>Single-Point Crossover</strong>: Randomly select a ‘crossover point’ which should be smaller than the total 
length of the chromosome. Take the two parents and swap the gene values between them ONLY for 
those genes which appear AFTER the crossover point to create two new children. </br>

<strong>Mutation </strong>: This is dependent on your representation, look at the lecture slides for some ideas on which 
mutation to implement given your representation. Your mutation function must take a single integer 
parameter which will determine how many times it is repeated on a solution (e.g., M(1) – one 
mutation per chromosome, M(3) – 3 mutations). </br>

<strong>Weakest Replacement </strong>: If the new solution is fitter than the worst in the population, then overwrite the 
worst (breaking ties randomly) with the new solution. </br>

<strong>Implementation and Experimentation </strong>
Implement the described EA in such a way that you can address the above problem and then run the 
experiments described below and answer the subsequent questions. Note that, in all of the below, a 
single trial means that you run the algorithm once and stop it when 10,000 fitness evaluations have 
been reached. Different trials of the same algorithm should be seeded with different random number 
seeds. 

You should devise your own set of experiments to determine what effect (if any) the following 
parameters have on the performance of the optimisation: 
1. Tournament size(t)
2. Population size (p)
3. Mutation rate (i.e. the parameter identified in the mutation operator above) (m) </br>
Your experiments should assess the performance of the algorithm over a number of randomly seeded 
trials for each setting of t, p, m, to provide robust results. </br>

<strong>Analysis</strong><br/>
Record the best fitness reached at the end of each trial and any other variables during the run that you 
think will be important in answering the following questions. 

<strong>Hint</strong>: You should think carefully about how best to present your results to show the behaviour of the 
algorithm during your trials </br>

Question 1: Which combination of parameters produces the best results? </br>
Question 2: Why do you think this is the case? </br>
Question 3: What was the effect when you removed mutation? What about crossover? </br>
Question 4: If you were to extend your EA to work with a multi-objective version of this problem, </br>
which functions in your program would you change and why? </br>

In your answers, describe your observations of the results, and describe any tentative explanations or 
conclusions you feel like making, in addition to any further experiments you felt it interesting or useful 
to do to answer the above questions or to further your understanding of the algorithm parameters
