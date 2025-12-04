import numpy as np
from bert_score import score as score_bert
from scipy.optimize import linear_sum_assignment
from typing import Dict

from evaluate.construction.common.graph.utils import split_to_edges
from evaluate.construction.common.graph.io import load_lines_safe


def get_bert_score(all_gold_edges, all_pred_edges):
	"""
	그래프를 엣지 문장 집합으로 보고, BERTScore F1을 비용 행렬로 하여 헝가리안 매칭.
	"""
	references = []
	candidates = []

	ref_cand_index = {}
	for (gold_edges, pred_edges) in zip(all_gold_edges, all_pred_edges):
		for (i, gold_edge) in enumerate(gold_edges):
			for (j, pred_edge) in enumerate(pred_edges):
				references.append(gold_edge)
				candidates.append(pred_edge)
				ref_cand_index[(gold_edge, pred_edge)] = len(references) - 1

	_, _, bs_F1 = score_bert(cands=candidates, refs=references, lang='en', idf=False)
	print("Computed bert scores for all pairs")

	precisions, recalls, f1s = [], [], []
	for (gold_edges, pred_edges) in zip(all_gold_edges, all_pred_edges):
		score_matrix = np.zeros((len(gold_edges), len(pred_edges)))
		for (i, gold_edge) in enumerate(gold_edges):
			for (j, pred_edge) in enumerate(pred_edges):
				score_matrix[i][j] = bs_F1[ref_cand_index[(gold_edge, pred_edge)]]

		row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

		sample_precision = score_matrix[row_ind, col_ind].sum() / len(pred_edges)
		sample_recall = score_matrix[row_ind, col_ind].sum() / len(gold_edges)

		precisions.append(sample_precision)
		recalls.append(sample_recall)
		f1s.append(2 * sample_precision * sample_recall / (sample_precision + sample_recall))

	return np.array(precisions), np.array(recalls), np.array(f1s)


def g_bertcore(pred_path: str, gold_path: str) -> Dict[str, float]:
	"""
	파일 경로를 입력받아 G-BERTScore를 계산하여 평균 P/R/F1을 반환한다.
	"""
	gold_graphs = load_lines_safe(gold_path)
	pred_graphs = load_lines_safe(pred_path)

	if len(gold_graphs) != len(pred_graphs):
		raise ValueError("gold와 pred의 샘플 수가 일치하지 않습니다.")

	gold_edges = split_to_edges(gold_graphs)
	pred_edges = split_to_edges(pred_graphs)

	precisions, recalls, f1s = get_bert_score(gold_edges, pred_edges)
	n = len(gold_graphs) if len(gold_graphs) > 0 else 1
	return {
		"precision": float(precisions.sum()) / float(n),
		"recall": float(recalls.sum()) / float(n),
		"f1": float(f1s.sum()) / float(n),
	}


