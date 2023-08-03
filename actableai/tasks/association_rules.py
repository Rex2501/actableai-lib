from io import StringIO
from itertools import product
import time
from typing import Dict, List
import pandas as pd

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAIAssociationRulesTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.ASSOCIATION_RULES)
    def run(
        self,
        df: pd.DataFrame,
        group_by: List[str],
        items: str,
        frequent_method: str = "fpgrowth",
        min_support: float = 0.5,
        association_metric: str = "confidence",
        min_association_metric: float = 0.5,
        graph_top_k: int = 10,
    ) -> Dict:
        """Generate association rules from a dataframe.

        Args:
            df: Input dataframe.
            group_by: List of columns to group by. (e.g. order_id or customer_id)
            items: Column name of items. (e.g. product_id or product_name)
            frequent_method: Frequent method to use. Available options
                are ["fpgrowth", "fpmax", "apriori"]
            min_support: Minimum support threshold for itemsets generation.
            association_metric: Association metric used for association rules
                generation. Available options are
                ["support", "confidence", "lift", "leverage", "conviction"]
            min_association_metric: Minimum value for significance of association.
            graph_top_k: Maximum number of nodes to display on association graph.

        Examples:
            >>> import pandas as pd
            >>> from actableai.tasks.association_rules import AssociationRulesTask
            >>> df = pd.read_csv("path/to/data.csv")
            >>> result = AssociationRulesTask().run(
            ...     df,
            ...     group_by=["order_id", "customer_id"],
            ...     items="product_id",
            ... )
            >>> result["association_rules"]

        Returns:
            Dict: Dictionnary containing the results of the task.
                - "status": "SUCCESS" if the task successfully ran else "FAILURE"
                - "data": Dictionnary containing the data of the task.
                    - "rules": List of association rules.
                    - "frequent_itemset": Frequent itemsets.
                    - "df_list": List of associated items for each group_by.
                    - "graph": Association graph.
                    - "association_metric": Association metric used for association
                        rules generation.
                    - "association_rules_chord": Association rules chord diagram.
                - "validations": List of validations on the data,
                    non-empty if the data presents a problem for the task
                - "runtime": Time taken to run the task.
        """

        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import (
            apriori,
            association_rules,
            fpgrowth,
            fpmax,
        )
        import networkx as nx

        from actableai.data_validation.checkers import NoFrequentItemSet
        from actableai.data_validation.params import AssociationRulesDataValidator
        from actableai.data_validation.base import CheckLevels

        # Check parameters
        assert frequent_method in [
            "fpgrowth",
            "apriori",
            "fpmax",
        ], f"frequent_method must be one of 'fpgrowth', 'apriori' or 'fpmax'. Got {frequent_method}."
        assert association_metric in [
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ], f"frequent_method must be one of 'support', 'confidence', 'lift', 'leverage', 'conviction'. Got {association_metric}."

        start = time.time()
        df = df.copy()

        # Validate parameters
        data_validation_results = AssociationRulesDataValidator().validate(
            df, group_by, items
        )
        failed_checks = [
            check for check in data_validation_results if check is not None
        ]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "data": {},
                "validations": [
                    {"name": check.name, "level": check.level, "message": check.message}
                    for check in failed_checks
                ],
                "runtime": time.time() - start,
            }

        # Data Imputation
        df = df.fillna("empty")
        df = df.astype(str)

        # Encode the data
        df_list = df.groupby(group_by)[items].apply(list).reset_index()
        te = TransactionEncoder()
        df_encoded = te.fit_transform(df_list[items])
        df_encoded = pd.DataFrame(
            df_encoded,
            columns=te.columns_,
            index=df_list[group_by].astype(str).apply(", ".join, axis=1),
        )

        # Run the association rule analysis
        frequent_dict = {
            "fpgrowth": fpgrowth,
            "apriori": apriori,
            "fpmax": fpmax,
        }
        frequent_func = frequent_dict[frequent_method]
        frequent_itemset = frequent_func(
            df_encoded,
            min_support=min_support,
            use_colnames=True,
            max_len=None,
            verbose=1,
        )
        check_no_frequent_itemset = NoFrequentItemSet(level=CheckLevels.CRITICAL).check(
            frequent_itemset
        )
        if check_no_frequent_itemset is not None:
            return {
                "status": "FAILURE",
                "data": {},
                "validations": [
                    {
                        "name": check_no_frequent_itemset.name,
                        "level": check_no_frequent_itemset.level,
                        "message": check_no_frequent_itemset.message,
                    }
                ],
                "runtime": time.time() - start,
            }
        try:
            rules = association_rules(
                frequent_itemset,
                metric=association_metric,
                min_threshold=min_association_metric,
            )
        # When choosing fpmax as frequent_method, it discards the itemsets
        # that are contained in other itemsets.
        # This behavior of discarding itemsets makes it impossible to compute other
        # metrics than the current support which raises an error.
        except KeyError:
            association_metric = "support"
            rules = association_rules(
                frequent_itemset,
                metric=association_metric,
                min_threshold=0,
                support_only=True,
            )

        # Preparing a symetrical matrix like d3.chords calls for
        temp_df = rules.copy()
        temp_df["weight"] = (
            temp_df[association_metric] - temp_df[association_metric].min()
        ) / (temp_df[association_metric].max() - temp_df[association_metric].min())
        temp_df = temp_df[["antecedents", "consequents", association_metric]]
        nodes = list(set(temp_df["antecedents"]) | set(temp_df["consequents"]))
        matrix = {}
        for source, target in product(nodes, nodes):
            matrix[(source, target)] = 0
        for source, target, value in temp_df.to_records(index=False):
            matrix[(source, target)] = value
        m = [[matrix[(n1, n2)] for n1 in nodes] for n2 in nodes]
        association_rules_chord = {
            "nodes": list([",".join(x) for x in nodes]),
            "matrix": m,
        }

        temp_df = rules.copy().head(graph_top_k)
        rules = rules.sort_values(by=association_metric, ascending=False)
        temp_df["antecedents"] = temp_df["antecedents"].apply(list).apply(", ".join)
        temp_df["consequents"] = temp_df["consequents"].apply(list).apply(", ".join)
        temp_df["weight"] = (
            temp_df[association_metric] - temp_df[association_metric].min()
        ) / (temp_df[association_metric].max() - temp_df[association_metric].min())
        temp_df["penwidth"] = temp_df["weight"] + 0.2
        # temp_df["arrowsize"] = temp_df["weight"]
        buffer = StringIO()
        graph = nx.from_pandas_edgelist(
            temp_df,
            source="antecedents",
            target="consequents",
            edge_attr=["penwidth", "weight"],
            create_using=nx.DiGraph(),
        )
        nx.drawing.nx_pydot.write_dot(graph, buffer)  # type: ignore

        return {
            "status": "SUCCESS",
            "data": {
                "rules": rules,
                "frequent_itemset": frequent_itemset,
                "df_list": df_list,
                "graph": buffer.getvalue(),
                "association_metric": association_metric,
                "association_rules_chord": association_rules_chord,
            },
            "validations": [],
            "runtime": time.time() - start,
        }
