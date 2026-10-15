"""Process-pool Math-Verify reward for miles — a hang-proof drop-in replacement
for math_verify_custom.compute_math_verify_rewards.

WHY THIS EXISTS
---------------
math-verify scores a response by parsing it to sympy and checking symbolic
equivalence. On pathological inputs sympy can spin in a C-level / GIL-holding
computation that NO in-process timeout can interrupt: signal.alarm only fires
between Python bytecodes (never inside a C call) and only on the main thread, and
a worker *thread* cannot be killed. Because miles computes rewards inline on the
rollout's single asyncio event loop, one such stuck call freezes the entire
rollout — generation stops, the GPUs go idle, the job wedges.

The only reliable way to bound runaway CPU work is to run it in a separate
PROCESS and kill the process when it overruns. This module does exactly that:

  * a persistent ProcessPoolExecutor (forkserver/spawn — never a bare fork from
    this multi-threaded actor) runs each verification in a child;
  * the child runs on its own main thread, so math-verify's native signal.alarm
    timeouts work there as a first line of defence;
  * a verification RUNNING past the hard deadline is scored `timeout_score`
    (0.0) and the pool is force-killed, so a runaway sympy call can never
    linger or block anything;
  * the blocking wait is offloaded to a thread (run_in_executor), so the rollout
    event loop keeps dispatching generation while rewards compute.

THE COLD-START CASCADE (fixed 2026-07-16)
-----------------------------------------
An earlier version enforced the deadline on every future, running or not, and
killed the pool on any miss. On FIRST use each worker still has to import
sympy/math-verify; under startup CPU contention (all workers spawning while the
sglang engines saturate the node) those imports can exceed the deadline, so the
first samples were scored 0.0 and the pool was killed — and the next call
rebuilt a cold pool, timed out again, and killed it again. That cascade
silently zeroed SOME OR ALL rewards of a run's first eval/rollout (observed as
"step-0 baseline eval 0.0-0.18 with healthy generations"; wandb debug runs
step0dbg_*, 2026-07-16). Three changes make it impossible:

  * the pool is WARMED at build time: one import-forcing task per worker,
    waited on once (generously, no kill) before any real sample is scored;
  * the per-item deadline counts only time a future spends actually RUNNING in
    a worker; queued-not-started work (cold pool, busy workers) never trips it.
    A generous whole-batch cap still bounds the total wait;
  * every timeout, pool kill, and died-with-the-pool sample is LOGGED — the
    silent 0.0 was the truly harmful part.

Scoring is otherwise identical to math_verify_custom: math_metric with
gold=(LatexExtractionConfig,), pred=(ExprExtractionConfig, LatexExtractionConfig),
the label wrapped in \\boxed{...}, the full response as the prediction, and any
exception -> 0.0.

Wire in with:
  --custom-rm-path miles.rollout.rm_hub.math_verify_process.compute_math_verify_rewards

Tunables (CLI flag > env var > module default):
  --math-verify-num-workers / MATH_VERIFY_NUM_WORKERS  pool size            (default: 32)
  --math-verify-timeout-s   / MATH_VERIFY_TIMEOUT_S    per-sample hard kill (default: 10)
                              MATH_VERIFY_MP_START      start method         (default: forkserver)
                              MATH_VERIFY_WARMUP_TIMEOUT_S  one-time warmup wait (default: 120)
                              MATH_VERIFY_BATCH_GRACE_S     batch-cap slack      (default: 30)
"""

from __future__ import annotations

import asyncio
import logging
import math
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait as futures_wait

try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Please install math-verify: pip install math-verify") from exc

logger = logging.getLogger(__name__)

_TIMEOUT_SCORE = 0.0
# Defaults below are overridden by the --math-verify-* CLI flags (via _apply_args)
# on the first reward call, falling back to these env vars, then these literals.
_DEFAULT_NUM_WORKERS = 32
_PER_ITEM_TIMEOUT_S = float(os.environ.get("MATH_VERIFY_TIMEOUT_S", "10"))
_NUM_WORKERS = int(os.environ.get("MATH_VERIFY_NUM_WORKERS", str(_DEFAULT_NUM_WORKERS)))
_WARMUP_TIMEOUT_S = float(os.environ.get("MATH_VERIFY_WARMUP_TIMEOUT_S", "120"))
_BATCH_GRACE_S = float(os.environ.get("MATH_VERIFY_BATCH_GRACE_S", "30"))
_WEDGE_POLL_S = 0.5
_ARGS_APPLIED = False


def _start_method() -> str:
    # forkserver/spawn build workers from a clean interpreter, avoiding the
    # deadlocks of fork()ing this multi-threaded (asyncio + sglang) actor.
    want = os.environ.get("MATH_VERIFY_MP_START", "forkserver")
    available = mp.get_all_start_methods()
    if want in available:
        return want
    return "spawn" if "spawn" in available else available[0]


# --- worker side (runs in a child process, on its MAIN thread) ----------------
_VERIFY_FUNC = None


def _get_verify_func():
    global _VERIFY_FUNC
    if _VERIFY_FUNC is None:
        _VERIFY_FUNC = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    return _VERIFY_FUNC


def _warmup_worker() -> bool:
    """Force the heavy imports (sympy et al.) plus one end-to-end verification in
    this worker, so no real sample ever pays the import cost against its
    scoring deadline."""
    try:
        _get_verify_func()(["\\boxed{1}"], ["\\boxed{1}"])
    except Exception:
        pass
    return True


def _score_in_worker(response: str, ground_truth_unboxed: str) -> float:
    """Single verification. math-verify's native (signal.alarm) timeouts apply
    here since the child is on its main thread; the parent enforces the hard
    kill for anything those can't interrupt."""
    try:
        verify_func = _get_verify_func()
        gt_boxed = f"\\boxed{{{ground_truth_unboxed}}}"
        score, _ = verify_func([gt_boxed], [response])
        return float(score)
    except Exception:
        return 0.0


# --- parent side --------------------------------------------------------------
# Reward batches for different groups can run concurrently (each in its own
# run_in_executor thread); the lock serialises only pool create/warmup/teardown,
# never the scoring wait, so batches still score in parallel across the workers.
_POOL: ProcessPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _get_pool_locked() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        cpus = os.cpu_count() or 1
        if _NUM_WORKERS > cpus:
            logger.warning(
                f"math-verify: {_NUM_WORKERS} workers > {cpus} visible CPUs; "
                "worker startup/scoring will contend — consider --math-verify-num-workers <= CPUs"
            )
        t0 = time.monotonic()
        _POOL = ProcessPoolExecutor(
            max_workers=_NUM_WORKERS,
            mp_context=mp.get_context(_start_method()),
        )
        # Warm every worker before serving: a cold worker importing sympy under
        # startup CPU contention can exceed the scoring deadline, and a cold
        # start must never cost a sample its reward (see module docstring).
        # Callers queue on _POOL_LOCK meanwhile, which is exactly the intent.
        warmups = [_POOL.submit(_warmup_worker) for _ in range(_NUM_WORKERS)]
        done, not_done = futures_wait(warmups, timeout=_WARMUP_TIMEOUT_S)
        if not_done:
            logger.warning(
                f"math-verify: pool warmup incomplete after {_WARMUP_TIMEOUT_S:.0f}s "
                f"({len(done)}/{len(warmups)} warmup tasks done); continuing — remaining "
                "workers finish importing in the background (harmless: queued work "
                "doesn't count against the scoring deadline)"
            )
        else:
            logger.info(
                f"math-verify: pool of {_NUM_WORKERS} workers warm in {time.monotonic() - t0:.1f}s"
            )
    return _POOL


def _kill_pool_locked(expected: ProcessPoolExecutor) -> None:
    """Force-kill every worker and drop the pool so a runaway child can't linger.
    No-op if another thread already replaced the pool. A fresh pool is built
    (and warmed) lazily on the next call."""
    global _POOL
    if _POOL is not expected or _POOL is None:
        return
    pool, _POOL = _POOL, None
    for proc in list(getattr(pool, "_processes", {}).values()):
        try:
            proc.kill()  # SIGKILL: the only thing that stops uninterruptible C-level sympy
        except Exception:
            pass
    pool.shutdown(wait=False, cancel_futures=True)


def _score_batch(items: list[tuple[str, str]]) -> list[float]:
    """Score a batch in worker processes with a hard wall-clock deadline. Runs in
    a worker THREAD (via run_in_executor), so blocking here never stalls the
    rollout event loop."""
    if not items:
        return []

    with _POOL_LOCK:
        pool = _get_pool_locked()
        try:
            futures = [pool.submit(_score_in_worker, resp, label) for resp, label in items]
        except Exception:
            # pool was broken (e.g. workers previously killed) — rebuild once.
            logger.warning("math-verify: pool broken at submit; rebuilding")
            _kill_pool_locked(pool)
            pool = _get_pool_locked()
            futures = [pool.submit(_score_in_worker, resp, label) for resp, label in items]

    # The per-item deadline counts only time spent RUNNING in a worker: queued
    # work waiting behind a busy pool must neither be scored 0 nor kill the pool
    # (that was the cold-start cascade). The batch cap bounds the total wait so
    # a truly stuck pool still can't hold a batch forever.
    waves = math.ceil(len(items) / max(1, _NUM_WORKERS))
    batch_cap_s = _PER_ITEM_TIMEOUT_S * waves + _BATCH_GRACE_S
    batch_deadline = time.monotonic() + batch_cap_s
    running_since: dict = {}
    pending = set(futures)
    kill_reason = None
    while pending and kill_reason is None:
        _done, pending = futures_wait(pending, timeout=_WEDGE_POLL_S)
        now = time.monotonic()
        for fut in pending:
            if fut.running() and now - running_since.setdefault(fut, now) > _PER_ITEM_TIMEOUT_S:
                kill_reason = (
                    f"a verification ran past {_PER_ITEM_TIMEOUT_S:.0f}s in its worker "
                    "(wedged sympy?)"
                )
                break
        if kill_reason is None and pending and now > batch_deadline:
            kill_reason = f"batch overran its {batch_cap_s:.0f}s cap with {len(pending)} samples unscored"

    if kill_reason:
        # at least one worker is wedged in uninterruptible C-level work (or the
        # pool is starved beyond reason) — kill the whole pool so those processes
        # die and their CPUs are reclaimed.
        with _POOL_LOCK:
            _kill_pool_locked(pool)

    results: list[float] = []
    died_with_pool = 0
    for fut in futures:
        if fut.done():
            try:
                results.append(float(fut.result()))
            except Exception:
                # future finished with an infra error (pool killed mid-flight by
                # this or a concurrent batch), not a scoring result.
                died_with_pool += 1
                results.append(_TIMEOUT_SCORE)
        else:
            results.append(_TIMEOUT_SCORE)

    if kill_reason:
        unscored = sum(1 for fut in futures if not fut.done())
        logger.warning(
            f"math-verify: killed worker pool — {kill_reason}; "
            f"{unscored + died_with_pool}/{len(futures)} samples scored {_TIMEOUT_SCORE}"
        )
    elif died_with_pool:
        logger.warning(
            f"math-verify: {died_with_pool}/{len(futures)} samples scored {_TIMEOUT_SCORE} "
            "because the pool was killed by a concurrent batch"
        )

    return results


def _label_to_str(label) -> str:
    return "" if label is None else str(label)


def _apply_args(args) -> None:
    """Apply the --math-verify-* CLI flags (read off `args`) once, before the
    worker pool is first built. Precedence: CLI flag > env var > module default."""
    global _NUM_WORKERS, _PER_ITEM_TIMEOUT_S, _ARGS_APPLIED
    if _ARGS_APPLIED:
        return
    n = getattr(args, "math_verify_num_workers", None)
    if n:
        _NUM_WORKERS = int(n)
    t = getattr(args, "math_verify_timeout_s", None)
    if t:
        _PER_ITEM_TIMEOUT_S = float(t)
    _ARGS_APPLIED = True


async def compute_math_verify_rewards(args, samples, **kwargs):
    """Custom rm entrypoint. Accepts a single Sample or a list of Samples."""
    _apply_args(args)
    single = not isinstance(samples, (list, tuple))
    sample_list = [samples] if single else list(samples)
    items = [(s.response or "", _label_to_str(s.label)) for s in sample_list]

    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(None, _score_batch, items)
    return scores[0] if single else scores
