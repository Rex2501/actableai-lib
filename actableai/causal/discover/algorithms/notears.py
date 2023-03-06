import logging

from causalnex.structure.notears import from_pandas

from actableai.causal.discover.algorithms.commons.base_runner import CausalDiscoveryRunner, CausalGraph, ProgressCallback
from actableai.causal.discover.model.causal_discovery import CausalDiscoveryPayload


class NotearsPayload(CausalDiscoveryPayload):
    max_iter: int = 100


class NotearsRunner(CausalDiscoveryRunner):
    name = "Notears"

    def __init__(self, p: NotearsPayload, progress_callback: ProgressCallback = None):
        super().__init__(p, progress_callback)
        self._max_iter = p.max_iter

    def do_causal_discovery(self) -> CausalGraph:
        self._encode_categorical_as_integers()

        logging.info(f"Running NOTEARS with max_iter={self._max_iter}, h_tol=1e-8 and w_threshold=0.0")

        causal_graph = self._build_causal_graph(
            labeled_graph=from_pandas(
                X=self._prepared_data,
                max_iter=self._max_iter,
                h_tol=1e-8,
                # we can use w_threshold=0, since the weight
                # filtering will be applied in the frontend
                w_threshold=0.0,
                # nodes that are not allowed to be child of anyone do
                # not have incoming edges, so they are causes
                tabu_child_nodes=self._constraints.causes,
                # nodes that are not allowed to be parent of anyone do
                # not have outgoing edges, so they are effects
                tabu_parent_nodes=self._constraints.effects,
                tabu_edges=self._constraints.forbiddenRelationships,
            ),
            has_weights=True,
            has_confidence_values=False,
        )

        self._report_progress(100.0)

        return causal_graph
