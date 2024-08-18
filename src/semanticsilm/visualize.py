import os
from pyvis.network import Network
from llama_index.core import load_index_from_storage
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

PERSIST_DIR = "./persist"

def load_graph():
    storage_context = StorageContext.from_defaults(
        graph_store=SimpleGraphStore.from_persist_dir(PERSIST_DIR),
        index_store=SimpleIndexStore.from_persist_dir(PERSIST_DIR)
    )
    index = load_index_from_storage(storage_context)
    return index.get_networkx_graph()

def visualize_networkx(g, output_file='silmarillion_graph_networkx.png'):
    plt.figure(figsize=(20,20))
    pos = nx.spring_layout(g, k=0.5, iterations=50)
    nx.draw(g, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=8, font_weight='bold', 
            edge_color='gray', width=1, alpha=0.7)

    edge_labels = nx.get_edge_attributes(g, 'relationship')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6)

    plt.title("Silmarillion Knowledge Graph", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"NetworkX graph saved to {output_file}")

def visualize_plotly(g, output_file='silmarillion_graph_plotly.html'):
    pos = nx.spring_layout(g)

    edge_x = []
    edge_y = []
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[node for node in g.nodes()],
        textposition="top center"
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Silmarillion Knowledge Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Interactive Silmarillion Knowledge Graph",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.write_html(output_file)
    print(f"Plotly graph saved to {output_file}")

def main():
    g = load_graph()
    visualize_networkx(g)
    visualize_plotly(g)

if __name__ == "__main__":
    main()