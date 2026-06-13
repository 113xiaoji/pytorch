import argparse
import json
import os
import statistics
import subprocess
import sys
import time


def _make_stable_tree_model(depth, width):
    import torch

    class StableTree(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(width, width, bias=False)
                    for _ in range(depth)
                ]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return StableTree()


def _make_unsupported_property_model(width):
    import torch

    class Box:
        @property
        def scale(self):
            return 2

    class UnsupportedProperty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.box = Box()
            self.weight = torch.nn.Parameter(torch.ones(width))

        def forward(self, x):
            return x * self.weight * self.box.scale

    return UnsupportedProperty()


def _child_main(args):
    import torch
    from torch._C._dynamo import guards

    if args.case == "stable-tree":
        model = _make_stable_tree_model(args.depth, args.width)
    elif args.case == "unsupported-property":
        model = _make_unsupported_property_model(args.width)
    else:
        raise AssertionError(f"unknown case: {args.case}")

    opt_model = torch.compile(model, backend="eager", fullgraph=True)
    x = torch.randn(args.batch, args.width)

    for _ in range(args.warmup):
        opt_model(x)

    guards.reset_guard_lookup_stats()
    start_ns = time.perf_counter_ns()
    for _ in range(args.iters):
        opt_model(x)
    elapsed_ns = time.perf_counter_ns() - start_ns

    stats = guards.get_guard_lookup_stats()
    payload = {
        "case": args.case,
        "iters": args.iters,
        "call_avg_us": elapsed_ns / args.iters / 1000,
        "lookup_count": int(stats["lookup_count"]),
        "lookup_avg_us": _avg_us(stats, "lookup_total_ns"),
        "slow_guard_avg_us": _avg_us(stats, "slow_guard_ns"),
        "partial_enable": int(
            stats["guard_last_success_actual_partial_enable"]
        ),
        "partial_hit": int(stats["guard_last_success_actual_partial_hit"]),
        "partial_miss": int(stats["guard_last_success_actual_partial_miss"]),
        "partial_residual_fail": int(
            stats["guard_last_success_actual_partial_residual_fail"]
        ),
        "partial_unsupported": int(
            stats["guard_last_success_actual_partial_unsupported"]
        ),
        "partial_unsupported_cached": int(
            stats["guard_last_success_actual_partial_unsupported_cached"]
        ),
    }
    print(json.dumps(payload, sort_keys=True))


def _avg_us(stats, key):
    count = max(int(stats["lookup_count"]), 1)
    return int(stats.get(key, 0)) / count / 1000


def _run_child(args, case):
    env = os.environ.copy()
    env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
    env["TORCHDYNAMO_GUARD_LOOKUP_STATS"] = "1"
    output = subprocess.check_output(
        [
            sys.executable,
            __file__,
            "--child",
            "--case",
            case,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--depth",
            str(args.depth),
            "--width",
            str(args.width),
            "--batch",
            str(args.batch),
        ],
        env=env,
        text=True,
    )
    return json.loads(output.splitlines()[-1])


def _median(rows, key):
    return statistics.median(float(row[key]) for row in rows)


def _print_summary(rows):
    print(
        "case call_us lookup_us slow_guard_us lookup_count "
        "partial_enable partial_hit partial_miss "
        "partial_unsupported partial_unsupported_cached"
    )
    for case in sorted({row["case"] for row in rows}):
        case_rows = [row for row in rows if row["case"] == case]
        print(
            " ".join(
                [
                    case,
                    f"{_median(case_rows, 'call_avg_us'):.3f}",
                    f"{_median(case_rows, 'lookup_avg_us'):.3f}",
                    f"{_median(case_rows, 'slow_guard_avg_us'):.3f}",
                    str(int(_median(case_rows, "lookup_count"))),
                    str(int(_median(case_rows, "partial_enable"))),
                    str(int(_median(case_rows, "partial_hit"))),
                    str(int(_median(case_rows, "partial_miss"))),
                    str(int(_median(case_rows, "partial_unsupported"))),
                    str(int(_median(case_rows, "partial_unsupported_cached"))),
                ]
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Dynamo guard last-success memo behavior."
    )
    parser.add_argument(
        "--case",
        choices=("stable-tree", "unsupported-property", "all"),
        default="all",
    )
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    if args.child:
        _child_main(args)
        return

    cases = (
        ("stable-tree", "unsupported-property")
        if args.case == "all"
        else (args.case,)
    )
    rows = [
        _run_child(args, case)
        for case in cases
        for _ in range(args.repeats)
    ]
    _print_summary(rows)
    print("\nraw_json:")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
