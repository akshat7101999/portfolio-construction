# Basic libraries
import os
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class GeneticAlgoStrategy:
	"""
	My own custom implementation of genetic algorithms for portfolio
	"""
	def __init__(self):
		print("Genetic algo strategy has been created")
		self.initial_genes = 100
		self.selection_top = 25
		self.mutation_iterations = 50
		self.weight_update_factor = 0.1
		self.gene_length = None
		self.genes_in_each_iteration = 250
		self.iterations = 50
		self.crossover_probability = 0.05

	# def generate_portfolio(self, symbols, return_matrix):
	# 	self.gene_length = len(symbols)

	# 	# Create initial genes
	# 	initial_genes = self.generate_initial_genes(symbols)

	# 	for i in range(self.iterations):
	# 		# Select
	# 		top_genes = self.select(return_matrix, initial_genes)
	# 		#print("Iteration %d Best Sharpe Ratio: %.3f" % (i, top_genes[0][0]))
	# 		top_genes = [item[1] for item in top_genes]

	# 		# Mutate
	# 		mutated_genes = self.mutate(top_genes)
	# 		initial_genes = mutated_genes

	# 	top_genes = self.select(return_matrix, initial_genes)
	# 	best_gene = top_genes[0][1]
	# 	transposed_gene = np.array(best_gene).transpose() # Gene is a distribution of weights for different stocks
	# 	return_matrix_transposed = return_matrix.transpose()
	# 	returns = np.dot(return_matrix_transposed, transposed_gene)
	# 	returns_cumsum = np.cumsum(returns)
		
	# 	ga_portfolio_weights = best_gene
	# 	ga_portfolio_weights = dict([(symbols[x], ga_portfolio_weights[x]) for x in range(0, len(ga_portfolio_weights))])
	# 	return ga_portfolio_weights

	def generate_portfolio(self, symbols, return_matrix):
		"""
		Calculates portfolio weights using a Genetic Algorithm, 
		clips negative weights to 0, and then renormalizes the remaining 
		positive weights to sum to 1.
		
		This function assumes that self.generate_initial_genes, 
		self.select, self.mutate, and self.iterations are defined 
		elsewhere in the class.
		"""
		self.gene_length = len(symbols)

		# 1. Genetic Algorithm Optimization
		
		# Create initial genes
		initial_genes = self.generate_initial_genes(symbols)

		for i in range(self.iterations):
			# Select
			top_genes = self.select(return_matrix, initial_genes)
			#print("Iteration %d Best Sharpe Ratio: %.3f" % (i, top_genes[0][0]))
			top_genes = [item[1] for item in top_genes]

			# Mutate
			mutated_genes = self.mutate(top_genes)
			initial_genes = mutated_genes

		top_genes = self.select(return_matrix, initial_genes)
		# The best_gene contains the raw portfolio weights (can contain negatives)
		best_gene = top_genes[0][1] 
		
		# The rest of the return/cumulative return calculation is not needed 
		# for the portfolio weights dictionary, but kept for context.
		transposed_gene = np.array(best_gene).transpose()
		return_matrix_transposed = return_matrix.transpose()
		returns = np.dot(return_matrix_transposed, transposed_gene)
		returns_cumsum = np.cumsum(returns)
		
		# --- Start of Clipping and Renormalization Logic ---

		# 2. Apply Clipping and Renormalization to the best_gene
		
		# a. Clip negative weights to 0
		# The best_gene is a list of floats, so convert it to a NumPy array for easy manipulation
		raw_weights = np.array(best_gene)
		clipped_weights = np.maximum(raw_weights, 0) 
		
		# b. Calculate the sum of the remaining positive weights
		sum_of_positives = np.sum(clipped_weights)
		
		# c. Renormalize the positive weights
		if sum_of_positives == 0:
			# If all weights were 0 or negative, the result is a zero portfolio
			final_weights = np.zeros_like(clipped_weights)
		else:
			# Divide each clipped positive weight by the new sum to make them sum to 1
			final_weights = clipped_weights / sum_of_positives
		
		# --- End of Clipping and Renormalization Logic ---

		# 3. Create the Final Dictionary
		# Use the renormalized weights
		ga_portfolio_weights = dict([(symbols[x], final_weights[x]) for x in range(0, len(final_weights))])
		
		return ga_portfolio_weights

	def generate_initial_genes(self, symbols):
		total_symbols = len(symbols)

		genes = []
		for i in range(self.initial_genes):
			gene = [random.uniform(-1, 1) for _ in range(0, total_symbols)]
			genes.append(gene)

		return genes

	def mutate(self, genes):
		new_genes = []

		for gene in genes: 
			for x in range(0, self.mutation_iterations):
				mutation = gene + (self.weight_update_factor * np.random.uniform(-1, 1, self.gene_length))
				mutation = list(mutation)
				new_genes.append(mutation)

		new_genes = genes + new_genes 
		random.shuffle(new_genes)
		genes_to_keep = new_genes[:self.genes_in_each_iteration] 

		# Add crossovers
		crossovers = self.crossover(new_genes)
		genes_to_keep = genes_to_keep + crossovers

		return genes_to_keep

	def select(self, return_matrix, genes):
		genes_with_scores = []
		for gene in genes:
			transposed_gene = np.array(gene).transpose() # Gene is a distribution of weights for different stocks
			return_matrix_transposed = return_matrix.transpose()
			returns = np.dot(return_matrix_transposed, transposed_gene)
			returns_cumsum = np.cumsum(returns)
			
			# Get fitness score
			fitness = self.fitness_score(returns)
			genes_with_scores.append([fitness, gene])
		
		# Sort
		random_genes = [self.generate_a_gene() for _ in range(5)]
		genes_with_scores = list(reversed(sorted(genes_with_scores)))
		genes_with_scores = genes_with_scores[:self.selection_top] + random_genes
		return genes_with_scores

	def fitness_score(self, returns):
		sharpe_returns = np.mean(returns) / np.std(returns)
		return sharpe_returns

	def generate_a_gene(self):
		gene = [random.uniform(-1, 1) for _ in range(self.gene_length)]
		return gene

	def crossover(self, population):
		crossover_population = []
		for z in range(0, len(population)):
			if random.uniform(0, 1) < self.crossover_probability:
				try:
					random_gene_first = list(random.sample(population, 1)[0])
					random_gene_second = list(random.sample(population, 1)[0])
					random_split = random.randrange(1, len(random_gene_first) - 1)
					crossover_gene = random_gene_first[:random_split] + random_gene_second[random_split:]
					crossover_population.append(crossover_gene)
				except Exception as e:
					continue

		return crossover_population