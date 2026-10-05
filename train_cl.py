import sys
import time
import random
import torch
import torch.nn.functional as F

from model.world_model import WorldModel
from data.gridworld_dynamics import make_world, make_sequence, VOCAB_SIZE


WORLDS = [42, 999, 2024]
LENGTH = 32
N_REPLAY_BATCHES = 20


def eval_on_world(model, world, device):
    model.eval()
    with torch.no_grad():
        seq, target = make_sequence(world, LENGTH, 128, device)
        logits = model(seq)
        pred = logits[:, :-1].argmax(-1)
        tgt = target[:, 1:]
        mask = tgt != -100
        return (pred[mask] == tgt[mask]).float().mean().item()


def train_on_world(model, world, steps, opt, device,
                   replay_buffer=None, replay_ratio=0.3, log_every=500):
    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        model.train()
        seq, target = make_sequence(world, LENGTH, 32, device)

        if replay_buffer and random.random() < replay_ratio:
            seq_b, target_b = random.choice(replay_buffer)
            seq = torch.cat([seq, seq_b.to(device)], dim=0)
            target = torch.cat([target, target_b.to(device)], dim=0)

        logits = model(seq)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE),
            target[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"    step {step}/{steps}  loss {avg:.4f}  elapsed {time.time()-t0:.0f}s")

    return time.time() - t0


def run_cl(method="baseline", steps_per_world=1000):
    device = "cuda"
    torch.manual_seed(0)
    model = WorldModel(vocab_size=VOCAB_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    replay_buffer = [] if method == "replay" else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"method={method}  params={n_params:,}  steps_per_world={steps_per_world}")

    for stage_idx, seed in enumerate(WORLDS):
        print(f"\n=== Stage {stage_idx}: 训练 world seed={seed} ===")
        world = make_world(seed=seed)
        t = train_on_world(model, world, steps_per_world, opt, device,
                           replay_buffer=replay_buffer)
        print(f"  训练耗时: {t:.0f}s")

        print(f"  --- 训练完 world {stage_idx} 后 ---")
        for eval_idx, eval_seed in enumerate(WORLDS[:stage_idx + 1]):
            eval_world = make_world(seed=eval_seed)
            acc = eval_on_world(model, eval_world, device)
            marker = "  ← 忘了吗？" if eval_idx < stage_idx else ""
            print(f"    world {eval_idx} (seed={eval_seed}): acc={acc:.4f}{marker}")

        if replay_buffer is not None and stage_idx < len(WORLDS) - 1:
            for _ in range(N_REPLAY_BATCHES):
                seq, target = make_sequence(world, LENGTH, 32, "cpu")
                replay_buffer.append((seq, target))
            print(f"  replay buffer size: {len(replay_buffer)}")

    # 保存
    torch.save(model.state_dict(), f"/kaggle/working/cl_{method}.pt")


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run_cl(method=method, steps_per_world=steps)