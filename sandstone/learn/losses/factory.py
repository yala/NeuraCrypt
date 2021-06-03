LOSS_REGISTRY = {}

NO_LOSS_ERR = 'Loss {} not in LOSS_REGISTRY! Available losses are {}'

def RegisterLoss(loss_name):
    """Registers a loss."""

    def decorator(f):
        LOSS_REGISTRY[loss_name] = f
        return f

    return decorator

def get_loss(loss_name):
    """Get loss from LOSS_REGISTRY based on loss_name."""

    if not loss_name in LOSS_REGISTRY:
        raise Exception(NO_LOSS_ERR.format(
            loss_name, LOSS_REGISTRY.keys()))

    loss = LOSS_REGISTRY[loss_name]

    return loss