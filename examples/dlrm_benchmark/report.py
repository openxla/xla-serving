# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregates `loadgen.py` result JSONs into Markdown summary tables.

Each result file is one `(arm, request_batch_size, concurrency|qps, repeat)`
measurement. When a `<name>.monitor.json` sits next to `<name>.json`, its server
CPU, memory, and GPU metrics are merged into the row.

Usage:
  report --results_dir=/work/results [--output=/work/results/REPORT.md]
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import glob
import itertools
import json
import math
import os
import statistics
from typing import Any, Callable, Sequence

from absl import app
from absl import flags


@dataclasses.dataclass(frozen=True)
class _Flags:
  """Command-line flags; only registered when run as a binary."""

  results_dir: flags.FlagHolder[str | None]
  output: flags.FlagHolder[str | None]
  baseline_arm: flags.FlagHolder[str]
  p99_budget_ms: flags.FlagHolder[float]


def _define_flags() -> _Flags:
  """Registers flags lazily so tests can import `report` alongside other tools."""
  return _Flags(
      results_dir=flags.DEFINE_string(
          'results_dir', None, 'Directory of JSONs.'
      ),
      output=flags.DEFINE_string(
          'output', None, 'Markdown file (default stdout).'
      ),
      baseline_arm=flags.DEFINE_string(
          'baseline_arm',
          'py',
          'Arm used as denominator for speedup columns.',
      ),
      p99_budget_ms=flags.DEFINE_float(
          'p99_budget_ms',
          20.0,
          'Tail-latency budget in ms (p99 <= budget) for the latency-bounded '
          'headline table; <= 0 disables that section.',
      ),
  )

_ARM_ORDER = [
    'xse-batch',
    'xse-nobatch',
    'py-mp-batch',
    'py-mp',
    'py-batch',
    'py',
]
_ARM_LABELS = {
    'xse-batch': 'XSE (batching)',
    'xse-nobatch': 'XSE (no batching)',
    'py-mp-batch': 'Python ×N processes + dyn. batching',
    'py-mp': 'Python ×N processes',
    'py-batch': 'Python + dyn. batching',
    'py': 'Python',
}


def _is_xse(arm: str) -> bool:
  return arm.startswith('xse')


def _label(arm: str) -> str:
  return _ARM_LABELS.get(arm, arm)


def _load(results_dir: str) -> list[dict[str, Any]]:
  """Reads every loadgen result JSON under `results_dir` (+ monitor data)."""
  rows = []
  for path in sorted(
      glob.glob(os.path.join(results_dir, '**', '*.json'), recursive=True)
  ):
    if path.endswith('.monitor.json') or path.endswith('.parity.json'):
      continue
    with open(path) as f:
      try:
        r = json.load(f)
      except json.JSONDecodeError:
        continue
    if 'latency_ms' not in r:
      continue
    mon_path = path[: -len('.json')] + '.monitor.json'
    if os.path.exists(mon_path):
      with open(mon_path) as f:
        r['monitor'] = json.load(f)
    r['_path'] = path
    r.setdefault('labels', {})
    rows.append(r)
  return rows


def _median(values: Sequence[float | None]) -> float:
  clean = [v for v in values if v is not None and not math.isnan(v)]
  return statistics.median(clean) if clean else float('nan')


def _fmt(v: float, digits: int = 1) -> str:
  if v is None or (isinstance(v, float) and math.isnan(v)):
    return '–'
  return f'{v:,.{digits}f}'


def _arm_sort_key(arm: str) -> tuple[int, str]:
  return (_ARM_ORDER.index(arm) if arm in _ARM_ORDER else 99, arm)


def _errors_cell(v: dict[str, Any], include_drops: bool) -> str:
  """`, N err (CODE n, ...)` for a table cell, or '' when there were none."""
  drops = v['drops'] if include_drops else 0
  count = v['errors'] + drops
  if not count:
    return ''
  parts = [f'{code} {n}' for code, n in v['error_codes'].items()]
  if drops:
    parts.append(f'client drops {drops}')
  return f', {count} err' + (f" ({', '.join(parts)})" if parts else '')


def _aggregate(
    rows: list[dict[str, Any]],
) -> dict[Any, dict[str, dict[str, Any]]]:
  """Groups repeats: key -> arm -> aggregated metrics (median over repeats)."""
  groups = collections.defaultdict(lambda: collections.defaultdict(list))
  for r in rows:
    labels = r['labels']
    platform = labels.get('platform', '?')
    emb = r.get('embedding_size', labels.get('embedding_size', '?'))
    arm = labels.get('arm', '?')
    if r['mode'] == 'closed_loop':
      load = ('C', r['concurrency'])
    elif 'peak_fraction' in labels:
      # run_benchmark.sh drives every arm at a fraction of its *own* peak, so
      # the comparable quantity across arms is the fraction, not the QPS.
      load = ('peak_fraction', float(labels['peak_fraction']))
    else:
      load = ('qps', r.get('target_qps'))
    key = (platform, emb, r['mode'], r['request_batch_size'], load)
    groups[key][arm].append(r)

  agg: dict[Any, dict[str, dict[str, Any]]] = {}
  for key, arms in groups.items():
    agg[key] = {}
    for arm, reps in arms.items():
      mon = [x.get('monitor', {}) for x in reps]
      ex_s = [x['examples_per_s'] for x in reps]
      agg[key][arm] = {
          'n': len(reps),
          'qps': _median([x['qps'] for x in reps]),
          'examples_per_s': _median(ex_s),
          'examples_per_s_cov': (
              statistics.pstdev(ex_s) / statistics.mean(ex_s)
              if len(ex_s) > 1 and statistics.mean(ex_s) > 0
              else 0.0
          ),
          'p50': _median([x['latency_ms']['p50'] for x in reps]),
          'p90': _median([x['latency_ms']['p90'] for x in reps]),
          'p99': _median([x['latency_ms']['p99'] for x in reps]),
          'max': _median([x['latency_ms']['max'] for x in reps]),
          'errors': sum(x['errors'] for x in reps),
          'error_codes': dict(
              sum(
                  (collections.Counter(x.get('error_codes', {})) for x in reps),
                  collections.Counter(),
              ).most_common()
          ),
          'drops': sum(x.get('overload_drops', 0) for x in reps),
          'server_cores': _median(
              [m.get('avg_cpu_cores', float('nan')) for m in mon]
          ),
          # Average busy cores over the monitored window divided by the
          # steady-state throughput: independent of how much warmup / worker
          # start-up time the monitor also covered.
          'server_cpu_s_per_1k_ex': _median([
              m['avg_cpu_cores'] / x['examples_per_s'] * 1000
              if m.get('avg_cpu_cores') is not None and x['examples_per_s'] > 0
              else float('nan')
              for m, x in zip(mon, reps)
          ]),
          'peak_rss_mb': _median(
              [m.get('peak_rss_mb', float('nan')) for m in mon]
          ),
          'gpu_util': _median(
              [m.get('gpu', {}).get('util_avg_pct', float('nan')) for m in mon]
          ),
      }
  return agg


def _render_headline_table(
    agg: dict[Any, dict[str, dict[str, Any]]],
    *,
    arms: Sequence[str],
    platforms: Sequence[str],
    embs: Sequence[Any],
    baseline_arm: str,
    point_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> list[str]:
  """Renders a summary table picking each arm's best closed-loop point."""
  xse_arms = [a for a in arms if _is_xse(a)]
  py_arms = [a for a in arms if not _is_xse(a)]
  has_speedup = baseline_arm in arms
  has_best_py = bool(xse_arms) and bool(py_arms)

  lines: list[str] = []
  lines.append(
      '| Platform | emb | B_req | '
      + ' | '.join(f'{_label(a)} ex/s (p99 ms)' for a in arms)
      + (f' | Speedup vs {baseline_arm}' if has_speedup else '')
      + (' | XSE vs best Python' if has_best_py else '')
      + ' |'
  )
  lines.append(
      '|' + '---|' * (3 + len(arms) + int(has_speedup) + int(has_best_py))
  )
  for platform in platforms:
    for emb in embs:
      breqs = sorted({
          k[3]
          for k in agg
          if k[0] == platform and k[1] == emb and k[2] == 'closed_loop'
      })
      for breq in breqs:
        best_by_arm: dict[str, dict[str, Any]] = {}
        for arm in arms:
          best = None
          for k, v in agg.items():
            if k[:4] == (platform, emb, 'closed_loop', breq) and arm in v:
              cand = v[arm]
              if point_filter is not None and not point_filter(cand):
                continue
              if (
                  best is None
                  or cand['examples_per_s'] > best['examples_per_s']
              ):
                best = cand
          if best is not None:
            best_by_arm[arm] = best

        cells = []
        for arm in arms:
          b = best_by_arm.get(arm)
          cells.append(
              f"{_fmt(b['examples_per_s'], 0)} ({_fmt(b['p99'])})" if b else '–'
          )

        speed = ''
        if has_speedup:
          base_row = best_by_arm.get(baseline_arm)
          base = base_row['examples_per_s'] if base_row is not None else 0.0
          if base > 0:
            speed_parts = [
                f"{a}: {best_by_arm[a]['examples_per_s'] / base:.2f}x"
                for a in arms
                if a != baseline_arm and a in best_by_arm
            ]
            speed = ' | ' + (', '.join(speed_parts) if speed_parts else '–')
          else:
            speed = ' | –'

        best_py = ''
        if has_best_py:
          candidates = [
              (best_by_arm[a]['examples_per_s'], a)
              for a in py_arms
              if a in best_by_arm
          ]
          if candidates and max(candidates)[0] > 0:
            best_rate, best_arm = max(candidates)
            ratios = [
                f"{a}: {best_by_arm[a]['examples_per_s'] / best_rate:.2f}x"
                for a in xse_arms
                if a in best_by_arm
            ]
            best_py = (
                ' | ' + ', '.join(ratios) + f' (best: {best_arm})'
                if ratios
                else ' | –'
            )
          else:
            best_py = ' | –'

        lines.append(
            f'| {platform} | {emb} | {breq} | '
            + ' | '.join(cells)
            + speed
            + best_py
            + ' |'
        )
  return lines


def render(
    rows: list[dict[str, Any]],
    baseline_arm: str = 'py',
    p99_budget_ms: float = 20.0,
) -> str:
  """Renders the Markdown report for the given result rows."""
  agg = _aggregate(rows)
  out: list[str] = []
  out.append(
      '# XSE vs. pure-Python JAX serving — DLRM-DCNv2 (JCE preprocessor)\n'
  )
  if not agg:
    out.append('_No results found._\n')
    return '\n'.join(out)

  arms = sorted(
      set(itertools.chain.from_iterable(agg.values())), key=_arm_sort_key
  )
  platforms = sorted({k[0] for k in agg})
  embs = sorted({k[1] for k in agg}, key=str)

  xse_arms = [a for a in arms if _is_xse(a)]
  py_arms = [a for a in arms if not _is_xse(a)]
  has_best_py = bool(xse_arms) and bool(py_arms)

  # ---- Headline 1: throughput under tail-latency budget --------------------
  if p99_budget_ms > 0:
    budget_str = f'{p99_budget_ms:g}'
    out.append(
        f'## Throughput under p99 ≤ {budget_str} ms latency budget '
        '(closed loop, median over repeats)\n'
    )
    suffix = '.'
    if has_best_py:
      suffix = (
          '; "XSE vs best Python" divides each XSE arm by the '
          'highest-throughput Python arm in that row.'
      )
    out.append(
        'Highest error-free examples/s among concurrency levels whose median '
        f'p99 latency stayed at or below {budget_str} ms{suffix}\n'
    )
    out.extend(
        _render_headline_table(
            agg,
            arms=arms,
            platforms=platforms,
            embs=embs,
            baseline_arm=baseline_arm,
            point_filter=lambda v: (
                v['errors'] == 0
                and not math.isnan(v['p99'])
                and v['p99'] <= p99_budget_ms
            ),
        )
    )
    out.append('')

  # ---- Headline 2: unbounded peak examples/s per arm -----------------------
  out.append(
      '## Peak throughput (closed loop, no latency limit, median over'
      ' repeats)\n'
  )
  if has_best_py:
    out.append(
        '"XSE vs best Python" divides each XSE arm by the highest-throughput '
        'Python arm in that row across single-process, dynamically batched, '
        'and multi-process configurations.\n'
    )
  out.extend(
      _render_headline_table(
          agg,
          arms=arms,
          platforms=platforms,
          embs=embs,
          baseline_arm=baseline_arm,
      )
  )
  out.append('')

  # ---- Detailed closed-loop sweep -----------------------------------------
  out.append('## Closed-loop concurrency sweep\n')
  out.append(
      'Columns per arm: requests/s, examples/s, p50 / p99 latency (ms), '
      'server CPU-s per 1k examples, peak memory (MB; PSS summed over the '
      'process tree for multi-process servers, RSS otherwise), errors.\n'
  )
  for platform in platforms:
    for emb in embs:
      keys = sorted(
          [
              k
              for k in agg
              if k[0] == platform and k[1] == emb and k[2] == 'closed_loop'
          ],
          key=lambda k: (k[3], k[4][1]),
      )
      if not keys:
        continue
      out.append(f'### {platform}, embedding_size={emb}\n')
      out.append('| B_req | C | ' + ' | '.join(_label(a) for a in arms) + ' |')
      out.append('|' + '---|' * (2 + len(arms)))
      for k in keys:
        cells = []
        for arm in arms:
          v = agg[k].get(arm)
          if not v:
            cells.append('–')
            continue
          cells.append(
              f"{_fmt(v['qps'])} rps, **{_fmt(v['examples_per_s'], 0)} ex/s**, "
              f"{_fmt(v['p50'], 2)} / {_fmt(v['p99'], 2)} ms, "
              f"{_fmt(v['server_cpu_s_per_1k_ex'], 3)} cpu-s/1k, "
              f"{_fmt(v['peak_rss_mb'], 0)} MB"
              + _errors_cell(v, include_drops=False)
              + (f" (CoV {v['examples_per_s_cov']:.0%})" if v['n'] > 1 else '')
          )
        out.append(f'| {k[3]} | {k[4][1]} | ' + ' | '.join(cells) + ' |')
      out.append('')

  # ---- Open-loop latency under load ---------------------------------------
  open_keys = [k for k in agg if k[2] == 'open_loop']
  if open_keys:
    out.append('## Open-loop (Poisson) latency under load\n')
    out.append(
        'Columns per arm: achieved rps, p50 / p90 / p99 / max (ms),'
        ' errors+drops.\n'
    )
    for platform in platforms:
      for emb in embs:
        keys = sorted(
            [k for k in open_keys if k[0] == platform and k[1] == emb],
            key=lambda k: (k[3], k[4][1]),
        )
        if not keys:
          continue
        out.append(f'### {platform}, embedding_size={emb}\n')
        out.append(
            '| B_req | load | ' + ' | '.join(_label(a) for a in arms) + ' |'
        )
        out.append('|' + '---|' * (2 + len(arms)))
        for k in keys:
          cells = []
          for arm in arms:
            v = agg[k].get(arm)
            if not v:
              cells.append('–')
              continue
            cells.append(
                f"{_fmt(v['qps'])} rps, {_fmt(v['p50'], 2)} /"
                f" {_fmt(v['p90'], 2)} / **{_fmt(v['p99'], 2)}** /"
                f" {_fmt(v['max'], 1)} ms"
                + _errors_cell(v, include_drops=True)
            )
          load_kind, load_value = k[4]
          if load_kind == 'peak_fraction':
            load = f'{load_value * 100:.0f}% of own peak'
          else:
            load = f'{_fmt(load_value, 0)} rps'
          out.append(f'| {k[3]} | {load} | ' + ' | '.join(cells) + ' |')
        out.append('')

  # ---- Provenance ---------------------------------------------------------
  out.append('## Run metadata\n')
  hosts = sorted({r.get('client_host', '?') for r in rows})
  out.append(f'- Files: {len(rows)} result JSONs')
  out.append(f'- Client host(s): {", ".join(hosts)}')
  labels = collections.defaultdict(set)
  for r in rows:
    for k, v in r['labels'].items():
      labels[k].add(str(v))
  for k in sorted(labels):
    out.append(f'- {k}: {", ".join(sorted(labels[k]))}')
  return '\n'.join(out) + '\n'


def main(argv: Sequence[str], f: _Flags) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  if not f.results_dir.value:
    raise app.UsageError('--results_dir is required')
  text = render(
      _load(f.results_dir.value),
      baseline_arm=f.baseline_arm.value,
      p99_budget_ms=f.p99_budget_ms.value,
  )
  if f.output.value:
    os.makedirs(
        os.path.dirname(os.path.abspath(f.output.value)), exist_ok=True
    )
    with open(f.output.value, 'w') as out_f:
      out_f.write(text)
    print(f'Wrote {f.output.value}')
  else:
    print(text)


if __name__ == '__main__':
  app.run(functools.partial(main, f=_define_flags()))
