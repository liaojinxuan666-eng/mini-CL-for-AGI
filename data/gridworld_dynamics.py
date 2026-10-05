import torch

X_OFFSET = 0
Y_OFFSET = 100
A_OFFSET = 200
VOCAB_SIZE = 204
GRID_SIZE = 100
N_ACTIONS = 4

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def make_world(seed, grid_size=GRID_SIZE):
    g = torch.Generator().manual_seed(seed)
    rand = torch.rand(grid_size, grid_size, generator=g)
    world = torch.zeros(grid_size, grid_size, dtype=torch.long)
    world[rand >= 0.90] = 1
    world[rand >= 0.97] = 2
    return world


def step(world, x, y, a):
    t = world[x, y].item()
    dx, dy = ACTIONS[a]
    if t == 0:
        nx, ny = x + dx, y + dy
    elif t == 1:
        nx, ny = x - dx, y - dy
    else:
        nx = (x * 7 + 13) % GRID_SIZE
        ny = (y * 11 + 17) % GRID_SIZE
    nx = max(0, min(GRID_SIZE - 1, nx))
    ny = max(0, min(GRID_SIZE - 1, ny))
    return nx, ny


def make_sequence(world, length, batch_size, device, start_xy=None):
    B = batch_size
    xs = torch.zeros(B, length, dtype=torch.long)
    ys = torch.zeros(B, length, dtype=torch.long)
    as_ = torch.zeros(B, length, dtype=torch.long)

    for b in range(B):
        if start_xy is None:
            x = torch.randint(0, GRID_SIZE, (1,)).item()
            y = torch.randint(0, GRID_SIZE, (1,)).item()
        else:
            x, y = start_xy

        for t in range(length):
            xs[b, t] = x
            ys[b, t] = y
            a = torch.randint(0, N_ACTIONS, (1,)).item()
            as_[b, t] = a
            x, y = step(world, x, y, a)

    seq = torch.stack([xs, ys, as_], dim=-1).reshape(B, length * 3)
    seq[:, 0::3] += X_OFFSET
    seq[:, 1::3] += Y_OFFSET
    seq[:, 2::3] += A_OFFSET

    target = torch.full_like(seq, -100)
    target[:, 3::3] = seq[:, 3::3]
    target[:, 4::3] = seq[:, 4::3]

    return seq.to(device), target.to(device)