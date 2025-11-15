# Basic libraries
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class EigenPortfolioStrategy:
	def __init__(self):
		print("Eigen portfolio strategy has been created")
		
	# def generate_portfolio(self, symbols, covariance_matrix, eigen_portfolio_number):
	# 	"""
	# 	Inspired by: https://srome.github.io/Eigenvesting-I-Linear-Algebra-Can-Help-You-Choose-Your-Stock-Portfolio/
	# 	"""
	# 	eig_values, eig_vectors = np.linalg.eigh(covariance_matrix)
	# 	market_eigen_portfolio = eig_vectors[:,-1] / np.sum(eig_vectors[:,-1]) # We don't need this but in case someone wants to analyze
	# 	eigen_portfolio = eig_vectors[:,-eigen_portfolio_number] / np.sum(eig_vectors[:,-eigen_portfolio_number]) # This is a portfolio that is uncorrelated to market and still yields good returns

	# 	portfolio_weights_dictionary = dict([(symbols[x], eigen_portfolio[x]) for x in range(0, len(eigen_portfolio))])
	# 	print(portfolio_weights_dictionary)
	# 	return portfolio_weights_dictionary

	def generate_portfolio(self, symbols, covariance_matrix, eigen_portfolio_number):
		"""
		Inspired by: https://srome.github.io/Eigenvesting-I-Linear-Algebra-Can-Help-You-Choose-Your-Stock-Portfolio/
		
		Generates an eigen-portfolio, clips negative weights to 0, 
		and then renormalizes the remaining positive weights to sum to 1.
		"""
		# 1. Standard Eigen-Portfolio Calculation
		eig_values, eig_vectors = np.linalg.eigh(covariance_matrix)
		
		# Calculate the desired eigen-portfolio vector
		# Note: Initial eigen-portfolio weights typically sum to 1 (due to the division here), 
		# but they can contain negative values.
		eigen_portfolio = eig_vectors[:,-eigen_portfolio_number] / np.sum(eig_vectors[:,-eigen_portfolio_number]) 

		# 2. Apply Clipping and Renormalization
		
		# a. Clip negative weights to 0
		# Use np.maximum(array, 0) to replace any value < 0 with 0
		clipped_eigen_portfolio = np.maximum(eigen_portfolio, 0) 
		
		# b. Calculate the sum of the remaining positive weights
		sum_of_positives = np.sum(clipped_eigen_portfolio)
		
		# c. Renormalize the positive weights
		# If the sum is 0 (i.e., all original weights were 0 or negative), 
		# we return a dictionary of zeros to avoid division by zero.
		if sum_of_positives == 0:
			renormalized_portfolio = np.zeros_like(clipped_eigen_portfolio)
		else:
			# Divide each clipped positive weight by the new sum
			renormalized_portfolio = clipped_eigen_portfolio / sum_of_positives
		
		# 3. Create the Final Dictionary
		# Convert the resulting numpy array into the final symbol:weight dictionary
		portfolio_weights_dictionary = dict([(symbols[x], renormalized_portfolio[x]) for x in range(0, len(renormalized_portfolio))])
		
		print(portfolio_weights_dictionary)
		return portfolio_weights_dictionary
