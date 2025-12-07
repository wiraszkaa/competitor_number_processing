
import argparse
import yaml
from pathlib import Path
from .steps import (
    init_structure,
    step_collect,
    step_extract,
    step_candidates,
    step_patches,
    step_train,
    step_infer,
)

STEP_FUNS = {
    "collect": step_collect,
    "extract": step_extract,
    "candidates": step_candidates,
    "patches": step_patches,
    "train": step_train,
    "infer": step_infer,
}

def load_cfg(cfg_path: Path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def cmd_init(args):
    cfg = load_cfg(Path(args.config))
    init_structure(cfg, dry_run=args.dry_run)

def cmd_list(args):
    cfg = load_cfg(Path(args.config))
    steps = cfg.get("steps", [])
    for i, s in enumerate(steps, 1):
        print(f"{i}. {s['id']:>10} — {s.get('description','')}")

def cmd_run(args):
    cfg = load_cfg(Path(args.config))
    steps_cfg = cfg.get("steps", [])
    wanted = None
    if args.steps:
        wanted = [s.strip() for s in args.steps.split(",")]
    order = [s["id"] for s in steps_cfg if (not wanted or s["id"] in wanted)]
    for sid in order:
        fn = STEP_FUNS.get(sid)
        if fn is None:
            print(f"[WARN] brak implementacji kroku: {sid}")
            continue
        print(f"[PIPELINE] >>> {sid}")
        fn(cfg, dry_run=args.dry_run)

def main():
    ap = argparse.ArgumentParser(prog="pipeline", description="Minimalny pipeline (szkielet)")
    ap.set_defaults(func=None)
    ap.add_argument("--config", default="pipeline.yaml")
    sub = ap.add_subparsers()

    p_init = sub.add_parser("init")
    p_init.add_argument("--dry-run", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_list = sub.add_parser("list-steps")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run")
    p_run.add_argument("--steps", default=None, help="np. collect,extract,train")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    if args.func is None:
        ap.print_help()
    else:
        args.func(args)

if __name__ == "__main__":
    main()
