def create_evaluation(evaluation_type, **kwargs):
    from .multi_view_depth_evaluation import MultiViewDepthEvaluation
    from .robust_mvd_benchmark import RobustMultiViewDepthBenchmark

    if evaluation_type == 'mvd':
        return MultiViewDepthEvaluation(**kwargs)
    if evaluation_type == 'robustmvd':
        return RobustMultiViewDepthBenchmark(**kwargs)


def list_evaluations():
    return ['mvd', 'robustmvd']
