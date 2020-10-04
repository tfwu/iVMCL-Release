from .AOG import get_aog


def build_aog(cfg):
    """ Build AOGs

    Args:
        cfg (dict): cfg should contain:
            args: args needed to instantiate aogs.

    Returns:
        aogs:
    """
    assert isinstance(cfg, dict)
    cfg_ = cfg.copy()

    aogs = []

    dims = cfg_.pop('dims')
    max_splits = cfg_.pop('max_splits')
    extra_node_hierarchy = cfg_.pop('extra_node_hierarchy')
    remove_symmetric_children_of_or_node = cfg_.pop(
        'remove_symmetric_children_of_or_node')

    grid_ht = 1
    for i, grid_wd in enumerate(dims):
        aogs.append(
            get_aog(
                grid_ht=grid_ht,
                grid_wd=grid_wd,
                max_split=max_splits[i],
                use_tnode_topdown_connection=True \
                    if '1' in extra_node_hierarchy[i] else False,
                use_tnode_bottomup_connection_layerwise=True \
                    if '2' in extra_node_hierarchy[i] else False,
                use_tnode_bottomup_connection_sequential=True
                    if '3' in extra_node_hierarchy[i] else False,
                use_node_lateral_connection=True
                    if '4' in extra_node_hierarchy[i] else False,
                use_tnode_bottomup_connection=True
                    if '5' in extra_node_hierarchy[i] else False,
                use_node_lateral_connection_1=True
                    if '6' in extra_node_hierarchy[i] else False,
                remove_symmetric_children_of_or_node=\
                    remove_symmetric_children_of_or_node[i]
                ))

    return aogs
