from typing import List, Optional, Set, Tuple

from ray.data._internal.logical.interfaces import (
    LogicalOperator,
    LogicalOperatorSupportsProjectionPushdown,
    LogicalPlan,
    Rule,
)
from ray.data._internal.logical.operators.map_operator import Project
from ray.data._internal.planner.plan_expression.expression_visitors import (
    _ColumnReferenceCollector,
    _ColumnRewriter,
)
from ray.data.expressions import (
    AliasExpr,
    ColumnExpr,
    Expr,
    StarExpr,
)


def _collect_referenced_columns(exprs: List[Expr]) -> Optional[Set[str]]:
    """
    Extract all column names referenced by the given expressions.

    Recursively traverses expression trees to find all ColumnExpr nodes
    and collects their names.

    Example: For expression "col1 + col2", returns {"col1", "col2"}
    """
    # If any expression is star(), we need all columns
    if any(isinstance(expr, StarExpr) for expr in exprs):
        # TODO (goutam): Instead of using None to refer to All columns, resolve the AST against the schema.
        # https://github.com/ray-project/ray/issues/57720
        return None

    collector = _ColumnReferenceCollector()
    for expr in exprs or []:
        collector.visit(expr)
    return collector.referenced_columns


def _extract_simple_rename(expr: Expr) -> Optional[Tuple[str, str]]:
    """
    Check if an expression is a simple column rename.

    Returns (source_name, dest_name) if the expression is of form:
        col("source").alias("dest")
    where source != dest.

    Returns None for other expression types.
    """
    if isinstance(expr, AliasExpr) and isinstance(expr.expr, ColumnExpr):
        dest_name = expr.name
        source_name = expr.expr.name
        if source_name != dest_name:
            return source_name, dest_name
    return None


def _compose_projects(
    upstream_project: Project,
    downstream_project: Project,
    upstream_has_star: bool,
) -> List[Expr]:
    """
    Compose two Projects when the downstream has star().

    Strategy:
    - Emit a single star() only if the upstream had star() as well.
    - Evaluate upstream non-star expressions first, then downstream non-star expressions.
      With sequential projection evaluation, downstream expressions can reference
      upstream outputs without explicit rewriting.
    - Rename-of-computed columns will be dropped from final output by the evaluator
      when there's no later explicit mention of the source name.
    """
    return upstream_project.exprs + downstream_project.exprs


def _try_fuse_consecutive_projects(
    upstream_project: Project, downstream_project: Project
) -> Optional[Project]:
    """
    Attempt to merge two consecutive Project operations into one.

    Example: Upstream: [star(), col("x").alias("y")], Downstream: [star(), (col("y") + 1).alias("z")] → Fused: [star(), (col("x") + 1).alias("z")]
    """
    upstream_has_star: bool = upstream_project.has_star_expr()
    downstream_has_star: bool = downstream_project.has_star_expr()

    rewritten_exprs: List[Expr] = []
    # Intersection case: This is when downstream is a selection (no star), and we need to recursively rewrite the downstream expressions into the upstream column definitions.
    # Example: Upstream: [col("a").alias("b")], Downstream: [col("b").alias("c")] → Rewritten: [col("a").alias("c")]
    if not downstream_has_star:
        upstream_column_definitions = {
            expr.name: expr
            for expr in upstream_project.exprs
            if not isinstance(expr, StarExpr)
        }
        for expr in downstream_project.exprs:
            rewritten = _ColumnRewriter(upstream_column_definitions).visit(expr)
            rewritten_exprs.append(rewritten)
    else:
        # Composition case: downstream has star(), and we need to merge both upstream and downstream expressions.
        # Example:
        # Upstream: [star(), col("a").alias("b")], Downstream: [star(), col("b").alias("c")] → Rewritten: [star(), col("a").alias("b"), col("b").alias("c")]
        rewritten_exprs = _compose_projects(
            upstream_project,
            downstream_project,
            upstream_has_star,
        )

    return Project(
        upstream_project.input_dependency,
        exprs=rewritten_exprs,
        ray_remote_args=downstream_project._ray_remote_args,
    )


class ProjectionPushdown(Rule):
    """
    Optimization rule that pushes projections (column selections) down the query plan.

    This rule performs two optimizations:
    1. Fuses consecutive Project operations to eliminate redundant projections
    2. Pushes projections into data sources (e.g., Read operations) to enable
       column pruning at the storage layer
    """

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply projection pushdown optimization to the entire plan."""
        dag = plan.dag

        new_dag = dag._apply_transform(self._try_fuse_projects)
        new_dag = new_dag._apply_transform(self._try_push_projection_into_read_op)

        return LogicalPlan(new_dag, plan.context) if dag is not new_dag else plan

    @classmethod
    def _try_fuse_projects(cls, op: LogicalOperator) -> LogicalOperator:
        """
        Optimize a single Project operator.

        Steps:
        1. Iteratively fuse with upstream Project operations
        2. Push the resulting projection into the data source if possible
        """
        if not isinstance(op, Project):
            return op

        # Step 1: Iteratively fuse with upstream Project operations
        current_project: Project = op

        if not isinstance(current_project.input_dependency, Project):
            return op

        upstream_project: Project = current_project.input_dependency  # type: ignore[assignment]
        return _try_fuse_consecutive_projects(upstream_project, current_project)

    @classmethod
    def _try_push_projection_into_read_op(cls, op: LogicalOperator) -> LogicalOperator:
        if not isinstance(op, Project):
            return op

        project: Project = op

        # Step 2: Push projection into the data source if supported
        input_op = project.input_dependency

        if (
            not project.has_star_expr()  # Must be a selection, not additive
            and isinstance(input_op, LogicalOperatorSupportsProjectionPushdown)
            and input_op.supports_projection_pushdown()
        ):
            from pprint import pprint

            pprint(f">>> [DBG] current project: {project.exprs}")

            required_columns = _collect_referenced_columns(list(project.exprs))

            if required_columns is not None:  # None means star() was present
                print(f"required columns: {required_columns}")
                optimized_source = input_op.apply_projection(list(required_columns))

                is_simple_selection = all(
                    isinstance(expr, ColumnExpr) for expr in project.exprs
                )

                if is_simple_selection:
                    # Simple column selection: Read handles everything
                    return optimized_source
                else:
                    # Has transformations: Keep Project on top of optimized Read
                    return Project(
                        optimized_source,
                        exprs=project.exprs,
                        ray_remote_args=project._ray_remote_args,
                    )

        return project
