# LaGraph architecture simplification plan

## Goal

Reduce module stacking without silently changing existing experimental results.

The default runtime profile remains `full`, which preserves the current training
and scoring path. Simplified variants are exposed as explicit ablation profiles.

## Runtime profiles

| Profile | Channel graph | Temporal graph | VQ bypass | Boundary detector | Multi-scale scorer | Intended use |
|---|---:|---:|---:|---:|---:|---|
| `full` | on | on | on | on | on | Reproduce current results |
| `core` | on | on | off | off | on | Main simplified candidate |
| `reconstruction` | off | off | off | off | off | Lower-bound reconstruction baseline |

## Recommendation

Use `full` as the current reference result. Run `core` next. If `core` keeps the
same ranking and stays within an acceptable metric drop, describe LaGraph as:

> decomposition + dual graph refinement + temporal encoder + multi-scale
> reconstruction scoring

Then treat VQ bypass and BoundaryDetector as optional diagnostic modules rather
than core architectural contributions.

## Commands

```powershell
python ts_benchmark/run_single.py --epochs 30 --arch-profile full
python ts_benchmark/run_single.py --epochs 30 --arch-profile core
python ts_benchmark/run_single.py --epochs 30 --arch-profile reconstruction
```

Keep batch size, learning rate, seeds, datasets, and threshold logic unchanged
when comparing these profiles.
