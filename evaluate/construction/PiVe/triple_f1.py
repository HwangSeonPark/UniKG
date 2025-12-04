from typing import Dict
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score

from evaluate.construction.common.graph.io import load_lines_safe
from evaluate.construction.common.graph.utils import modify_graph


def get_triple_match_f1(gold_graphs, pred_graphs):
	"""
	  트리플 전체 일치 기반 micro F1
	"""
	new_gold_graphs = [modify_graph(graph) for graph in gold_graphs]
	new_pred_graphs = [modify_graph(graph) for graph in pred_graphs]
	new_gold_graphs_list = [[str(string).lower() for string in sublist] for sublist in new_gold_graphs]
	new_pred_graphs_list = [[str(string).lower() for string in sublist] for sublist in new_pred_graphs]

	allclasses = new_pred_graphs_list + new_gold_graphs_list
	allclasses = [item for items in allclasses for item in items]
	allclasses = list(set(allclasses))

	lb = preprocessing.MultiLabelBinarizer(classes=allclasses)
	mcbin = lb.fit_transform(new_pred_graphs_list)
	mrbin = lb.fit_transform(new_gold_graphs_list)

	precision = precision_score(mrbin, mcbin, average='micro')
	recall = recall_score(mrbin, mcbin, average='micro')
	f1 = f1_score(mrbin, mcbin, average='micro')
	return float(f1), float(precision), float(recall)


def triple_f1(pred_path: str, gold_path: str) -> Dict[str, float]:
	"""
	파일 경로를 입력받아 Triple Match micro F1을 계산하여 반환한다.
	반환: {'f1': float, 'precision': float, 'recall': float}
	"""
	gold_graphs = load_lines_safe(gold_path)
	pred_graphs = load_lines_safe(pred_path)

	if len(gold_graphs) != len(pred_graphs):
		raise ValueError("gold와 pred의 샘플 수가 일치하지 않습니다.")

	f1, p, r = get_triple_match_f1(gold_graphs, pred_graphs)
	return {"f1": f1, "precision": p, "recall": r}


