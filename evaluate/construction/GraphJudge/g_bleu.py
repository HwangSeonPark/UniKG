import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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


def g_bleu(pred_path: str, gold_path: str) -> dict:
	"""
	G-BLEU score
	- Input file is a list of triples in each line
	- BLEU is calculated by sentence unit, and each edge is considered as a sentence
	- Matching is performed by the Hungarian algorithm on the BLEU score matrix
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

	import time
	try:
		from evaluate.construction.common import logger as log
	except:
		log = None
	
	precisions_bleu = []
	recalls_bleu = []
	f1s_bleu = []
	
	ntot = len(gold_tokens)
	tstr = time.time()
	if log:
		log.info(f"[G-BLEU] Evaluating {ntot} samples")

	for i in range(len(gold_tokens)):
		tcur = time.time() - tstr
		tavg = tcur / (i + 1) if i > 0 else 0
		tlft = tavg * (ntot - i - 1)
		pct = int(((i + 1) / ntot) * 100)
		
		if log and (i % max(1, ntot // 20) == 0 or i < 3):
			log.info(f"[G-BLEU] Sample {i+1}/{ntot} ({pct}%) | Elapsed: {tcur:.1f}s | Remaining: {tlft:.1f}s")
		
		score_bleu = np.zeros((len(pred_tokens[i]), len(gold_tokens[i])))
		for p_idx in range(len(pred_tokens[i])):
			for g_idx in range(len(gold_tokens[i])):
				score_bleu[p_idx, g_idx] = sentence_bleu(
					[gold_tokens[i][g_idx]],
					pred_tokens[i][p_idx],
					smoothing_function=SmoothingFunction().method1
				)
		p, r, f = _scores(score_bleu)
		precisions_bleu.append(p)
		recalls_bleu.append(r)
		f1s_bleu.append(f)
	
	print(f"  [G-BLEU] Evaluation completed: {ntot} samples")

	n = len(gold_graphs) if len(gold_graphs) > 0 else 1
	return {
		"precision": float(np.sum(precisions_bleu)) / float(n),
		"recall": float(np.sum(recalls_bleu)) / float(n),
		"f1": float(np.sum(f1s_bleu)) / float(n),
	}


