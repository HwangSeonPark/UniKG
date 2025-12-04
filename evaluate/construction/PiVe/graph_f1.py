import numpy as np
import networkx as nx
from typing import Dict

from evaluate.construction.common.graph.io import load_lines_safe
from evaluate.construction.common.graph.utils import return_eq_node, return_eq_edge


def get_graph_match_accuracy(pred_graphs, gold_graphs):
	"""

	두 그래프가 라벨 동치 조건으로 동형(isomorphic)인지 판별하여 비율 계산.
	- 노드/엣지 라벨 모두 동일해야 동형으로 간주
	"""
	matchs = 0
	for pred, gold in zip(pred_graphs, gold_graphs):
		g1 = nx.DiGraph()
		g2 = nx.DiGraph()

		for edge in gold:
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

		for edge in pred:
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

		# 노드/엣지 레이블 동치 조건 모두 적용
		if nx.is_isomorphic(g1, g2, node_match=return_eq_node, edge_match=return_eq_edge):
			matchs += 1
	acc = matchs/len(pred_graphs)
	return acc


def graph_f1(pred_path: str, gold_path: str) -> Dict[str, float]:
	"""
	파일 경로를 입력받아 그래프 매칭 정확도를 평균으로 산출해 반환한다.
	반환: {'accuracy': float}
	"""
	gold_graphs = load_lines_safe(gold_path)
	pred_graphs = load_lines_safe(pred_path)

	if len(gold_graphs) != len(pred_graphs):
		raise ValueError("gold와 pred의 샘플 수가 일치하지 않습니다.")

	acc = float(get_graph_match_accuracy(pred_graphs, gold_graphs))
	return {"accuracy": acc}


