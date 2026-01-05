import numpy as np
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment
from typing import Tuple

from evaluate.construction.common.graph.utils import split_to_edges, get_tokens
from evaluate.construction.common.graph.io import load_lines_safe


def _scores(cost_matrix: np.ndarray) -> Tuple[float, float, float]:
	row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
	precision = float(cost_matrix[row_ind, col_ind].sum()) / float(cost_matrix.shape[0]) if cost_matrix.shape[0] > 0 else 0.0
	recall = float(cost_matrix[row_ind, col_ind].sum()) / float(cost_matrix.shape[1]) if cost_matrix.shape[1] > 0 else 0.0
	f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
	return precision, recall, f1


def g_rouge(pred_path: str, gold_path: str) -> dict:
	"""
	G-ROUGE score
	- Input file is a list of triples in each line
	- Each edge is considered as a sentence, and matching is performed by the Hungarian algorithm on the ROUGE-2 precision matrix
	Return:
	- {'precision': float, 'recall': float, 'f1': float}
	"""
	gold_graphs = load_lines_safe(gold_path)
	pred_graphs = load_lines_safe(pred_path)

	if len(gold_graphs) != len(pred_graphs):
		raise ValueError("The number of samples in gold and pred do not match.")

	gold_edges = split_to_edges(gold_graphs)
	pred_edges = split_to_edges(pred_graphs)
	gold_tokens, pred_tokens = get_tokens(gold_edges, pred_edges)

	scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)

	precisions_rouge = []
	recalls_rouge = []
	f1s_rouge = []

	for i in range(len(gold_tokens)):
		# rouge2 precision 기반 점수 행렬
		score_rouge = np.zeros((len(pred_tokens[i]), len(gold_tokens[i])))
		for p_idx in range(len(pred_tokens[i])):
			for g_idx in range(len(gold_tokens[i])):
				score_rouge[p_idx, g_idx] = scorer_rouge.score(
					gold_edges[i][g_idx], pred_edges[i][p_idx]
				)['rouge2'].precision
		p, r, f = _scores(score_rouge)
		precisions_rouge.append(p)
		recalls_rouge.append(r)
		f1s_rouge.append(f)

	n = len(gold_graphs) if len(gold_graphs) > 0 else 1
	return {
		"precision": float(np.sum(precisions_rouge)) / float(n),
		"recall": float(np.sum(recalls_rouge)) / float(n),
		"f1": float(np.sum(f1s_rouge)) / float(n),
	}


