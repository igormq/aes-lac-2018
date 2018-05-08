import json

BLANK_TOKEN = '_'

def read_labels(labels_filepath, blank_label_id=0):
    """ Read labels file

    Args:
        labels_filepath: label file location
        blank_label_id: blank label id.
    """
    with open(labels_filepath, 'r', encoding='utf8') as f:
        labels = json.load(f)

    # Insert blank label
    if BLANK_TOKEN not in labels:
        labels.insert(blank_label_id, BLANK_TOKEN)
    else:
        # ensure that blank label is in the right index
        index = labels.index(BLANK_TOKEN)

        if index != blank_label_id:
            labels.pop(index)
            labels.insert(blank_label_id, BLANK_TOKEN)

    return labels

def write_labels(labels, filepath):
    """ Write labels to file
    """
    with open(filepath, 'w', encoding='utf8') as f:
        json.dump(labels, f, indent=4)