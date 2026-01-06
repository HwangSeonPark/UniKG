
import numpy as np
from bert_score import score as score_bert
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from typing import Dict

from evaluate.construction.common.graph.utils import split_to_edges
from evaluate.construction.common.graph.io import load_lines_safe


def _get_bert_score(all_gold_edges, all_pred_edges):
	
	references = []
	candidates = []

	ref_cand_index = {}
	for i in tqdm(range(len(all_gold_edges))):
		gold_edges = all_gold_edges[i]
		pred_edges = all_pred_edges[i]
		for (i, gold_edge) in enumerate(gold_edges):
			for (j, pred_edge) in enumerate(pred_edges):
				references.append(gold_edge)
				candidates.append(pred_edge)
				ref_cand_index[(gold_edge, pred_edge)] = len(references) - 1


	try:
		_, _, bs_F1 = score_bert(cands=candidates, refs=references, model_type="bert-base-uncased", lang='en', idf=False, device="cuda:1")
	except Exception:
		_, _, bs_F1 = score_bert(cands=candidates, refs=references, model_type="bert-base-uncased", lang='en', idf=False, device="cpu")

	precisions, recalls, f1s = [], [], []
	for i in tqdm(range(len(all_gold_edges))):
		gold_edges = all_gold_edges[i]
		pred_edges = all_pred_edges[i]
		score_matrix = np.zeros((len(gold_edges), len(pred_edges)))
		for (i, gold_edge) in enumerate(gold_edges):
			for (j, pred_edge) in enumerate(pred_edges):
				score_matrix[i][j] = bs_F1[ref_cand_index[(gold_edge, pred_edge)]]

		row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

		sample_precision = score_matrix[row_ind, col_ind].sum() / len(pred_edges) if len(pred_edges) > 0 else 0.0
		sample_recall = score_matrix[row_ind, col_ind].sum() / len(gold_edges) if len(gold_edges) > 0 else 0.0

		precisions.append(sample_precision)
		recalls.append(sample_recall)
		f1s.append(2 * sample_precision * sample_recall / (sample_precision + sample_recall) if (sample_precision + sample_recall) > 0 else 0.0)

	return np.array(precisions), np.array(recalls), np.array(f1s)


def g_bertscore(pred_path: str, gold_path: str) -> Dict[str, float]:
	"""
	G-BERTScore score
	- Input file is a list of triples in each line
	- Each edge is considered as a sentence, and matching is performed by the Hungarian algorithm on the BERTScore F1 matrix
	Return:
	- {'precision': float, 'recall': float, 'f1': float}
	"""
	gold_graphs = load_lines_safe(gold_path)
	pred_graphs = load_lines_safe(pred_path)

	if len(gold_graphs) != len(pred_graphs):
		raise ValueError("The number of samples in gold and pred do not match.")

	gold_edges = split_to_edges(gold_graphs)
	pred_edges = split_to_edges(pred_graphs)

	precisions, recalls, f1s = _get_bert_score(gold_edges, pred_edges)
	n = len(gold_graphs) if len(gold_graphs) > 0 else 1

	return {
		"precision": float(precisions.sum()) / float(n),
		"recall": float(recalls.sum()) / float(n),
		"f1": float(f1s.sum()) / float(n),
	}


