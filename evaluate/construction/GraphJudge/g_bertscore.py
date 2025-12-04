
# g_bertscore: model_type="bert-base-uncased", device="cuda:1"(실패 시 cpu), 빈 엣지일 때 0.0으로 안전 처리
# g_bertcore: 모델 타입/디바이스 기본값(영어 기본은 roberta-large), 안전 처리 없음
# 결과적으로 BERT 임베딩이 달라 매칭 비용 행렬이 달라지고(헝가리안 매칭 결과 변화), 평균 P/R/F1이 달라짐. 그래서 G-BERTCore가 더 높은 점수로 나왔다.
# 더 강한 백본(표현력↑)으로 엣지 문장 임베딩 품질이 높아 헝가리안 매칭 결과가 인간 판단과 상관이 높은 경향
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
	G-BERTScore(그래프-수준 BERTScore) 점수를 계산한다.
	- 입력 파일은 각 줄에 트리플 배열이 있는 형식
	- 각 엣지를 문장으로 보고, 엣지 간 BERTScore F1을 비용 행렬로 하여 헝가리안 매칭
	반환:
	- {'precision': float, 'recall': float, 'f1': float}
	"""
	gold_graphs = load_lines_safe(gold_path)
	pred_graphs = load_lines_safe(pred_path)

	if len(gold_graphs) != len(pred_graphs):
		raise ValueError("gold와 pred의 샘플 수가 일치하지 않습니다.")

	gold_edges = split_to_edges(gold_graphs)
	pred_edges = split_to_edges(pred_graphs)

	precisions, recalls, f1s = _get_bert_score(gold_edges, pred_edges)
	n = len(gold_graphs) if len(gold_graphs) > 0 else 1

	return {
		"precision": float(precisions.sum()) / float(n),
		"recall": float(recalls.sum()) / float(n),
		"f1": float(f1s.sum()) / float(n),
	}


