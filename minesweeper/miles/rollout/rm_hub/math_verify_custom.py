"""Math-Verify reward for miles, ported 1:1 from the verl `multi_thread` reward
manager used in ~/exploration (verl/workers/reward_manager/multi_thread_naive.py +
verl/utils/reward_score/math_verify.py).

Scoring is identical to that setup:
  * math_metric with gold=(LatexExtractionConfig,) and pred=(ExprExtractionConfig, LatexExtractionConfig)
  * the ground-truth label is wrapped in \\boxed{...} before verification
  * the full model response is passed as the prediction (no manual boxed extraction)
  * any exception -> 0.0, per-item timeout -> `timeout_score` (default 0.0)

Wire it in with:  --custom-rm-path miles.rollout.rm_hub.math_verify_custom.compute_math_verify_rewards

miles calls a custom rm either with a single Sample (async_rm) or a list of
Samples (batched_async_rm); this handles both. The work is synchronous and runs
on the rollout actor's main thread (matching how miles' own deepscaler/math rule
rewards run), so the SIGALRM-based per-item timeout below behaves exactly as in
the verl actors.
"""

from __future__ import annotations

import concurrent.futures
import signal

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Please install math-verify: pip install math-verify") from exc


# --- make math-verify usable off the main thread -----------------------------
# math-verify enforces its parse/verify timeouts with signal.alarm(), which only
# works on the main thread and otherwise raises
#   ValueError: signal only works in main thread of the main interpreter
# miles computes rewards inside a Ray async actor whose asyncio loop runs on a
# NON-main thread, so every verification raised and the broad `except Exception`
# below turned EVERY reward into 0.0 (silent: all rollout/eval rewards were 0).
# We swap math-verify's signal-based `timeout` for a thread-safe one (worker
# thread + Future.result timeout) that raises the same TimeoutException, so the
# behaviour is identical on the main thread and now also correct off it.
def _threadsafe_timeout(timeout_seconds: int | None = 10):
    if timeout_seconds is None or timeout_seconds <= 0:
        def _no_timeout(func):
            return func

        return _no_timeout

    def _decorator(func):
        def _wrapper(*args, **kwargs):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError as exc:
                raise TimeoutException("Operation timed out!") from exc
            finally:
                # Never block on a runaway sympy call (a thread can't be killed).
                executor.shutdown(wait=False)

        return _wrapper

    return _decorator


def _patch_math_verify_timeout():
    import math_verify.grader as _grader
    import math_verify.metric as _metric
    import math_verify.parser as _parser
    import math_verify.utils as _utils

    for _mod in (_utils, _parser, _metric, _grader):
        _mod.timeout = _threadsafe_timeout


_patch_math_verify_timeout()


# One verify function per process (math_metric is somewhat expensive to build).
_VERIFY_FUNC = None


def _get_verify_func():
    global _VERIFY_FUNC
    if _VERIFY_FUNC is None:
        _VERIFY_FUNC = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    return _VERIFY_FUNC


class _ItemTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ItemTimeout


def _compute_score(model_output: str, ground_truth_unboxed: str, timeout_score: float = 0.0,
                   per_item_timeout_s: int = 10) -> float:
    verify_func = _get_verify_func()
    gt_boxed = f"\\boxed{{{ground_truth_unboxed}}}"

    have_alarm = False
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(per_item_timeout_s)
        have_alarm = True
    except ValueError:
        # Not on the main thread -> rely on math-verify's own guards + try/except.
        have_alarm = False

    try:
        score, _ = verify_func([gt_boxed], [model_output])
        return float(score)
    except _ItemTimeout:
        return float(timeout_score)
    except TimeoutException:
        return float(timeout_score)
    except Exception:
        return 0.0
    finally:
        if have_alarm:
            try:
                signal.alarm(0)
            except Exception:
                pass


def _label_to_str(label) -> str:
    if label is None:
        return ""
    return str(label)


def _score_sample(sample) -> float:
    return _compute_score(sample.response or "", _label_to_str(sample.label))


async def compute_math_verify_rewards(args, samples, **kwargs):
    """Custom rm entrypoint. Accepts a single Sample or a list of Samples."""
    if isinstance(samples, (list, tuple)):
        return [_score_sample(s) for s in samples]
    return _score_sample(samples)
