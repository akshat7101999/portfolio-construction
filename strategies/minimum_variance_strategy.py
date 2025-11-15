# Basic libraries
import os
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class MinimumVarianceStrategy:
	def __init__(self):
		print("Minimum Variance strategy has been created")
		
	# def generate_portfolio(self, symbols, covariance_matrix):
	# 	"""
	# 	Inspired by: https://srome.github.io/Eigenvesting-II-Optimize-Your-Portfolio-With-Optimization/
	# 	"""
	# 	inverse_cov_matrix = np.linalg.pinv(covariance_matrix)
	# 	ones = np.ones(len(inverse_cov_matrix))
	# 	inverse_dot_ones = np.dot(inverse_cov_matrix, ones)
	# 	min_var_weights = inverse_dot_ones / np.dot( inverse_dot_ones, ones)
	# 	portfolio_weights_dictionary = dict([(symbols[x], min_var_weights[x]) for x in range(0, len(min_var_weights))])
	# 	return portfolio_weights_dictionary

	def generate_portfolio(self, symbols, covariance_matrix):
		"""
		Calculates the Minimum Variance (Min-Var) portfolio weights, 
		clips negative weights to 0, and then renormalizes the remaining 
		positive weights to sum to 1.
		"""
		# 1. Standard Min-Var Portfolio Calculation
		
		inverse_cov_matrix = np.linalg.pinv(covariance_matrix)
		ones = np.ones(len(inverse_cov_matrix))
		inverse_dot_ones = np.dot(inverse_cov_matrix, ones)
		
		# Calculate the initial Min-Var portfolio weights (can contain negatives)
		min_var_weights = inverse_dot_ones / np.dot( inverse_dot_ones, ones)

		# --- Start of Clipping and Renormalization Logic ---

		# 2. Apply Clipping and Renormalization
		
		# a. Clip negative weights to 0
		# Use np.maximum(array, 0) to replace any value < 0 with 0
		clipped_weights = np.maximum(min_var_weights, 0) 
		
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
		portfolio_weights_dictionary = dict([(symbols[x], final_weights[x]) for x in range(0, len(final_weights))])
		
		return portfolio_weights_dictionary
