import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def openai_compute(n_params, batch_size, training_steps):
    # given in PF/s (hence the / 24 / 3600)
    return 6 * n_params * batch_size * training_steps / 24 / 3600


def excluded_from_params(parameter: torch.nn.Parameter, vocab_size=-1):
    return vocab_size in parameter.shape


def non_emb_param_count(model: torch.nn.Module, vocab_size=-1):
    return sum(p.numel() for p in model.parameters() if not excluded_from_params(p, vocab_size))
