def format_loss_dict(loss_dict, prefix=None, suffix=None):
    """Format a loss dictionary by adding optional prefix and/or suffix to each key.
    for example, format_loss_dict({'loss1': 0.5, 'loss2': 0.3}, prefix='train_', suffix='_epoch1')
    would return {'train_loss1_epoch1': 0.5, 'train_loss2_epoch1': 0.3}
    """
    formatted_dict = {}
    for key, value in loss_dict.items():
        new_key = key
        if prefix:
            new_key = f"{prefix}{new_key}"
        if suffix:
            new_key = f"{new_key}{suffix}"
        formatted_dict[new_key] = value
    return formatted_dict
