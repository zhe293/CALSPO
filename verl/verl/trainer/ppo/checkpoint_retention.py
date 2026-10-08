"""Finish a full per-step checkpoint before removing the previous one."""
from __future__ import annotations

import os
from pathlib import Path
import shutil


def finalize_checkpoint(output_dir: str, step: int) -> None:
    root = Path(output_dir).resolve(strict=True)
    if step < 1 or root.stat().st_uid != os.getuid():
        raise RuntimeError('Unsafe checkpoint output directory or step')
    current = root / f'global_step_{step}'
    required = (
        'actor/model_world_size_1_rank_0.pt',
        'actor/optim_world_size_1_rank_0.pt',
        'actor/extra_state_world_size_1_rank_0.pt',
        'data.pt',
        'ema_stats.json',
    )
    if current.is_symlink() or not current.is_dir() or current.stat().st_uid != os.getuid():
        raise RuntimeError(f'Unsafe or missing new checkpoint: {current}')
    for name in required:
        path = current / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f'Incomplete new checkpoint: {path}')

    marker = root / 'latest_checkpointed_iteration.txt'
    temporary = root / f'.latest_checkpointed_iteration.{os.getpid()}.tmp'
    with temporary.open('w') as stream:
        stream.write(str(step))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, marker)

    previous = root / f'global_step_{step - 1}'
    if previous.exists():
        if (previous.is_symlink() or previous.resolve() != previous
                or previous.parent != root or previous.stat().st_uid != os.getuid()):
            raise RuntimeError(f'Unsafe previous checkpoint: {previous}')
        shutil.rmtree(previous)
    print(f'[CHECKPOINT_ROTATION] step={step} kept={current.name} '
          f'removed={previous.name if not previous.exists() and step > 1 else "none"}', flush=True)
