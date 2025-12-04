import numpy as np
import networkx as nx
from typing import Dict

from evaluate.construction.common.graph.io import load_lines_safe
from evaluate.construction.common.graph.utils import return_eq_node, return_eq_edge


def get_ged(gold_graph, pred_graph=None):
	"""

	정규화된 Graph Edit Distance(GED) 계산.
	"""
	g1 = nx.DiGraph()
	g2 = nx.DiGraph()

	for edge in gold_graph:
		# 빈 문자열 처리
		h = str(edge[0]).lower().strip()
		r = str(edge[1]).lower().strip()
		t = str(edge[2]).lower().strip()
		h = h if h else 'null'
		r = r if r else 'null'
		t = t if t else 'null'
		g1.add_node(h, label=h)
		g1.add_node(t, label=t)
		g1.add_edge(h, t, label=r)

	#   상한(정규화 상수) 정의 유지
	normalizing_constant = g1.number_of_nodes() + g1.number_of_edges() + 30

	if pred_graph is None:
		return 1

	for edge in pred_graph:
		if len(edge) == 2:
			edge.append('NULL')
		elif len(edge) == 1:
			edge.append('NULL')
			edge.append('NULL')
		# 빈 문자열 처리
		h = str(edge[0]).lower().strip()
		r = str(edge[1]).lower().strip()
		t = str(edge[2]).lower().strip()
		h = h if h else 'null'
		r = r if r else 'null'
		t = t if t else 'null'
		g2.add_node(h, label=h)
		g2.add_node(t, label=t)
		g2.add_edge(h, t, label=r)

	ged = nx.graph_edit_distance(g1, g2, node_match=return_eq_node, edge_match=return_eq_edge)
	assert ged <= normalizing_constant
	return ged / normalizing_constant


def ged(pred_path: str, gold_path: str) -> Dict[str, float]:
	"""
	파일 경로를 입력받아 GED를 샘플 평균으로 산출하여 반환한다.
	반환: {'ged': float}
	"""
	gold_graphs = load_lines_safe(gold_path)
	pred_graphs = load_lines_safe(pred_path)

	if len(gold_graphs) != len(pred_graphs):
		raise ValueError("gold와 pred의 샘플 수가 일치하지 않습니다.")

	geds = []
	for gold, pred in zip(gold_graphs, pred_graphs):
		geds.append(float(get_ged(gold, pred)))
	n = len(geds) if len(geds) > 0 else 1
	return {"ged": float(np.sum(geds)) / float(n)}


