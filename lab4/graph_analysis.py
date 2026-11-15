from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MATRIX_PATH = BASE_DIR / "adjacency_matrix.csv"
EDGE_TYPES_PATH = BASE_DIR / "edge_types.csv"
GRAPH_PATH = BASE_DIR / "graph.svg"


def load_graph() -> tuple[nx.DiGraph, pd.DataFrame]:
    matrix = pd.read_csv(MATRIX_PATH, index_col="node")
    nodes = list(matrix.index)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for source in nodes:
        for target in nodes:
            if matrix.at[source, target] > 0:
                graph.add_edge(source, target)
    edge_types = pd.read_csv(EDGE_TYPES_PATH)
    return graph, edge_types


def summarize(graph: nx.DiGraph) -> dict[str, object]:
    undirected = graph.to_undirected()
    degrees = {
        node: {"in": graph.in_degree(node), "out": graph.out_degree(node), "total": graph.degree(node)}
        for node in graph.nodes
    }
    betweenness = nx.betweenness_centrality(undirected, normalized=False)
    shortest = {
        "Исходное_видео": nx.shortest_path_length(undirected, "Исходное_видео", "Готовые_клипы")
    }
    component_sizes = {}
    for node in graph.nodes:
        g_copy = undirected.copy()
        g_copy.remove_node(node)
        if g_copy.number_of_nodes() == 0:
            component_sizes[node] = [0]
        else:
            component_sizes[node] = sorted((len(component) for component in nx.connected_components(g_copy)), reverse=True)
    critical_node = min(
        graph.nodes,
        key=lambda node: (
            max(component_sizes[node]),
            -betweenness.get(node, 0.0),
            -graph.degree(node),
        ),
    )
    return {
        "degrees": degrees,
        "betweenness": betweenness,
        "shortest_path": shortest,
        "component_sizes": component_sizes,
        "critical_node": critical_node,
    }


def draw_graph(graph: nx.DiGraph, edge_types: pd.DataFrame) -> None:
    edge_labels = {(row["from"], row["to"]): row["type"] for _, row in edge_types.iterrows()}
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="#dfe7ff",
        node_size=2200,
        edge_color="#3a3a3a",
        linewidths=1.2,
        font_size=9,
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(GRAPH_PATH, format="svg")
    plt.close()


def main() -> None:
    graph, edge_types = load_graph()
    summary = summarize(graph)
    draw_graph(graph, edge_types)

    print("Матрица смежности:")
    print(pd.read_csv(MATRIX_PATH, index_col="node"))
    print("\nСтепени вершин:")
    for node in graph.nodes:
        values = summary["degrees"][node]
        print(f"{node}: in={values['in']}, out={values['out']}, total={values['total']}")
    print("\nКритическая вершина:", summary["critical_node"])
    g_copy = graph.to_undirected().copy()
    g_copy.remove_node(summary["critical_node"])
    components = [sorted(component) for component in nx.connected_components(g_copy)]
    print("Компоненты после удаления критической вершины:", components)
    print("Размеры компонент:", sorted((len(component) for component in components), reverse=True))
    print("Кратчайший путь от исходного видео до готового клипа:", summary["shortest_path"])
    print("\nСамые важные вершины по betweenness centrality:")
    ordered = sorted(summary["betweenness"].items(), key=lambda item: item[1], reverse=True)
    for node, value in ordered:
        print(f"{node}: {value:.3f}")
    print(f"\nГраф сохранен в {GRAPH_PATH}")


if __name__ == "__main__":
    main()
