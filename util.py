def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


def flatten(list_to_flatten):
    result = []
    if isinstance(list_to_flatten, (list, tuple)):
        for x in list_to_flatten:
            result.extend(flatten(x))
    else:
        result.append(list_to_flatten)
    return result


def state_dict_to_list(state_dict):
    return flatten([x.tolist() for x in state_dict.values()])
