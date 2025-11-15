# Basic libraries
import os
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class MaximumSharpeRatioStrategy:
	def __init__(self):
		print("Maximum sharpe ratio strategy has been created")
		
	# def generate_portfolio(self, symbols, covariance_matrix, returns_vector):
	# 	"""
	# 	Inspired by: Eigen Portfolio Selection: A Robust Approach to Sharpe Ratio Maximization, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3070416
	# 	"""
	# 	inverse_cov_matrix = np.linalg.pinv(covariance_matrix)
	# 	ones = np.ones(len(inverse_cov_matrix))

	# 	numerator = np.dot(inverse_cov_matrix, returns_vector)
	# 	denominator = np.dot(np.dot(ones.transpose(), inverse_cov_matrix), returns_vector)
	# 	msr_portfolio_weights = numerator / denominator
		
	# 	portfolio_weights_dictionary = dict([(symbols[x], msr_portfolio_weights[x]) for x in range(0, len(msr_portfolio_weights))])
	# 	return portfolio_weights_dictionary



	def generate_portfolio(self, symbols, covariance_matrix, returns_vector):
		"""
		Calculates the Maximum Sharpe Ratio (MSR) portfolio weights, 
		clips negative weights to 0, and then renormalizes the remaining 
		positive weights to sum to 1.
		"""
		# 1. Standard MSR Portfolio Calculation
		# Note: Risk-free rate (r_f) is implicitly assumed to be 0 here, 
		# as the formula maximizes (w^T * mu) / sqrt(w^T * Sigma * w)
		# where w^T * mu is the excess return (mu - r_f)
		
		inverse_cov_matrix = np.linalg.pinv(covariance_matrix)
		ones = np.ones(len(inverse_cov_matrix))

		numerator = np.dot(inverse_cov_matrix, returns_vector)
		denominator = np.dot(np.dot(ones.transpose(), inverse_cov_matrix), returns_vector)
		
		# Calculate the initial MSR portfolio weights (can contain negatives)
		msr_portfolio_weights = numerator / denominator

		# --- Start of Clipping and Renormalization Logic ---

		# 2. Apply Clipping and Renormalization
		
		# a. Clip negative weights to 0
		# Use np.maximum(array, 0) to replace any value < 0 with 0
		clipped_weights = np.maximum(msr_portfolio_weights, 0) 
		
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
		
		# You might want to print the dictionary here for debugging, as in the previous example
		# print(portfolio_weights_dictionary) 
		
		return portfolio_weights_dictionary