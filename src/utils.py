def infer_roles(holds_indices, holds_xy):
    """Infer roles from holds_indices based on the hold IDs."""
    assert len(holds_indices) == len(holds_xy), "Holds indices and coordinates must match in length."
    roles = []
    # Find indices of non-foot holds (index<323) with smallest and largest y coordinate
    ys = [(index, holds_xy[i][1]) for (i,index) in enumerate(holds_indices) if index<323] # Only consider non-foot holds
    start_index = min(ys, key=lambda x: x[1])[0]  # Start hold (smallest y)
    end_index = max(ys, key=lambda x: x[1])[0]
    for idx, hold_index in enumerate(holds_indices):
        if hold_index == start_index:
            roles.append('s')  # Start hold
        elif hold_index == end_index:
            roles.append('e')  # End hold
        elif hold_index < 323:
            roles.append('m')  # Middle hold
        else:
            roles.append('f')  # Foot hold
    return roles