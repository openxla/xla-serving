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

"""Unit tests for report.py."""

import json
import math
import os
import tempfile

from absl.testing import absltest

from examples.dlrm_benchmark import report


def _closed_row(
    arm: str,
    *,
    breq: int = 1,
    concurrency: int = 16,
    exps: float,
    p50: float = 2.0,
    p99: float = 8.0,
    errors: int = 0,
    error_codes: dict[str, int] | None = None,
    overload_drops: int = 0,
    cores: float | None = 4.0,
    rss_mb: float | None = 8192.0,
) -> dict[str, object]:
  mon: dict[str, object] = {}
  if cores is not None:
    mon['avg_cpu_cores'] = cores
  if rss_mb is not None:
    mon['peak_rss_mb'] = rss_mb
  return {
      'mode': 'closed_loop',
      'request_batch_size': breq,
      'concurrency': concurrency,
      'qps': exps / breq,
      'examples_per_s': exps,
      'latency_ms': {
          'p50': p50,
          'p90': (p50 + p99) / 2,
          'p99': p99,
          'max': p99 * 1.5,
      },
      'errors': errors,
      'error_codes': error_codes or {},
      'overload_drops': overload_drops,
      'embedding_size': 128,
      'client_host': 'bench-vm',
      'labels': {'arm': arm, 'platform': 'tpu'},
      'monitor': mon,
  }


class ReportTest(absltest.TestCase):

  def test_empty_rows(self):
    md = report.render([])
    self.assertIn('_No results found._', md)

  def test_latency_budget_and_peak_tables_differ_above_budget(self):
    rows = [
        # xse-batch stays under 20 ms at both C=64 and C=256.
        _closed_row('xse-batch', concurrency=64, exps=18_000, p99=10.0),
        _closed_row('xse-batch', concurrency=256, exps=30_000, p99=16.4),
        # py-mp-batch stays <= 20 ms at C=64 (6k) and exceeds 20 ms at C=256.
        _closed_row('py-mp-batch', concurrency=64, exps=6_000, p99=8.6),
        _closed_row('py-mp-batch', concurrency=256, exps=9_000, p99=53.0),
        # py simple only reaches 750 under 20 ms.
        _closed_row('py', concurrency=4, exps=750, p99=5.4),
        _closed_row('py', concurrency=64, exps=830, p99=78.0),
    ]
    md = report.render(rows, baseline_arm='py', p99_budget_ms=20.0)
    self.assertIn('## Throughput under p99 ≤ 20 ms latency budget', md)
    self.assertIn('## Peak throughput (closed loop, no latency limit', md)

    sections = md.split('## ')
    budget_sec = next(s for s in sections if s.startswith('Throughput under'))
    peak_sec = next(s for s in sections if s.startswith('Peak throughput'))

    # Under 20 ms: 30k / 6k = 5.00x vs best Python, 30k / 750 = 40.00x vs py.
    self.assertIn('"XSE vs best Python" divides each XSE arm', budget_sec)
    self.assertIn('30,000 (16.4)', budget_sec)
    self.assertIn('6,000 (8.6)', budget_sec)
    self.assertIn('xse-batch: 5.00x (best: py-mp-batch)', budget_sec)
    self.assertIn('xse-batch: 40.00x', budget_sec)

    # Unbounded peak: 30,000 / 9,000 = 3.33x vs best Python.
    self.assertIn('"XSE vs best Python" divides each XSE arm', peak_sec)
    self.assertIn('9,000 (53.0)', peak_sec)
    self.assertIn('xse-batch: 3.33x (best: py-mp-batch)', peak_sec)

    # Markdown tables must be followed by a blank line before the next heading.
    self.assertIn('|\n\n## Peak throughput', md)
    self.assertIn('|\n\n## Closed-loop concurrency sweep', md)
    self.assertIn('|\n\n## Run metadata', md)
    self.assertIn('- Files: 6 result JSONs', md)
    self.assertIn('- Client host(s): bench-vm', md)
    self.assertIn('- arm: py, py-mp-batch, xse-batch', md)
    self.assertIn('- platform: tpu', md)

  def test_disable_latency_budget_and_error_filtering(self):
    rows = [
        # High-throughput point has errors -> excluded from budgeted table.
        # Closed-loop rows also omit `client drops` even if overload_drops > 0.
        _closed_row(
            'py-mp-batch',
            concurrency=16,
            exps=5_000,
            p99=12.0,
            errors=3,
            error_codes={'UNAVAILABLE': 3},
            overload_drops=2,
        ),
        _closed_row('xse-batch', concurrency=16, exps=20_000, p99=9.0),
    ]
    md = report.render(rows, p99_budget_ms=20.0)
    budget_sec = md.split('## ')[1]
    # py-mp-batch had no error-free point <= 20 ms, so its budgeted cell is '–'.
    self.assertIn('| tpu | 128 | 1 | 20,000 (9.0) | – | – |', budget_sec)
    # Detailed closed-loop table includes error code breakdown without drops.
    self.assertIn('3 err (UNAVAILABLE 3)', md)
    self.assertNotIn('client drops', md)

    md_no_budget = report.render(rows, p99_budget_ms=0.0)
    self.assertNotIn('Throughput under p99', md_no_budget)

  def test_load_merges_monitor_and_skips_parity_files(self):
    with tempfile.TemporaryDirectory() as tmp:
      arm_dir = os.path.join(tmp, 'xse-batch')
      os.makedirs(arm_dir)
      row = _closed_row('xse-batch', exps=10_000, p99=5.0)
      mon = row.pop('monitor')
      with open(os.path.join(arm_dir, 'closed_B1_C16_r1.json'), 'w') as f:
        json.dump(row, f)
      with open(
          os.path.join(arm_dir, 'closed_B1_C16_r1.monitor.json'), 'w'
      ) as f:
        json.dump(mon, f)
      with open(os.path.join(arm_dir, 'parity.parity.json'), 'w') as f:
        json.dump({'ok': True, 'latency_ms': {'p99': 1.0}}, f)

      loaded = report._load(tmp)
      self.assertLen(loaded, 1)
      md = report.render(loaded)
      # 4.0 cores / 10,000 ex/s * 1000 = 0.400 cpu-s/1k; 8192 MB.
      self.assertIn('0.400 cpu-s/1k, 8,192 MB', md)
      # XSE-only run omits the cross-family "XSE vs best Python" column.
      self.assertNotIn('XSE vs best Python', md)

  def test_open_loop_drops_nan_and_missing_baseline(self):
    nan = float('nan')
    rows = [
        {
            'mode': 'closed_loop',
            'request_batch_size': 16,
            'concurrency': 4,
            'qps': 0.0,
            'examples_per_s': 0.0,
            'latency_ms': {'p50': nan, 'p90': nan, 'p99': nan, 'max': nan},
            'errors': 5,
            'error_codes': {'DEADLINE_EXCEEDED': 5},
            'overload_drops': 0,
            'embedding_size': 128,
            'client_host': 'bench-vm',
            'labels': {'arm': 'xse-batch', 'platform': 'cpu'},
        },
        {
            'mode': 'open_loop',
            'request_batch_size': 16,
            'target_qps': 500.0,
            'qps': 498.2,
            'examples_per_s': 498.2 * 16,
            'latency_ms': {'p50': 2.1, 'p90': 4.5, 'p99': 9.8, 'max': 15.0},
            'errors': 1,
            'error_codes': {'UNAVAILABLE': 1},
            'overload_drops': 2,
            'embedding_size': 128,
            'client_host': 'bench-vm',
            'labels': {
                'arm': 'xse-batch',
                'platform': 'cpu',
                'peak_fraction': '0.8',
            },
        },
        {
            'mode': 'open_loop',
            'request_batch_size': 16,
            'target_qps': 200.0,
            'qps': 200.0,
            'examples_per_s': 3200.0,
            'latency_ms': {'p50': 1.5, 'p90': 3.0, 'p99': 5.0, 'max': 7.0},
            'errors': 1,
            'error_codes': {'DEADLINE_EXCEEDED': 1},
            'overload_drops': 0,
            'embedding_size': 128,
            'client_host': 'bench-vm',
            'labels': {'arm': 'xse-batch', 'platform': 'cpu'},
        },
    ]
    md = report.render(rows, baseline_arm='py', p99_budget_ms=20.0)
    self.assertIn('80% of own peak', md)
    self.assertIn('3 err (UNAVAILABLE 1, client drops 2)', md)
    self.assertIn('1 err (DEADLINE_EXCEEDED 1)', md)
    self.assertNotIn('client drops 0', md)
    self.assertIn('0 (–)', md)
    self.assertNotIn('XSE vs best Python', md)
    self.assertIn('|\n\n## Open-loop', md)
    self.assertIn('|\n\n## Run metadata', md)

  def test_median_filters_none_and_nan_and_python_only_omits_xse_column(self):
    nan = float('nan')
    self.assertEqual(report._median([None, nan, nan, 10.0]), 10.0)
    self.assertTrue(math.isnan(report._median([None, nan])))

    # Three repeats where two lack monitor metrics (NaN) and one succeeded:
    # median must ignore both NaNs and report the valid repeat.
    rows = [
        _closed_row('py', exps=500, p99=5.0, cores=None, rss_mb=None),
        _closed_row('py', exps=500, p99=5.0, cores=None, rss_mb=None),
        _closed_row('py', exps=500, p99=5.0, cores=2.0, rss_mb=4096.0),
        _closed_row('py-batch', exps=2_000, p99=8.0, cores=4.0, rss_mb=4096.0),
    ]
    md = report.render(rows, baseline_arm='py', p99_budget_ms=20.0)
    self.assertNotIn('XSE vs best Python', md)
    self.assertIn('Speedup vs py', md)
    self.assertIn('py-batch: 4.00x', md)
    # 2.0 cores / 500 ex/s * 1000 = 4.000 cpu-s/1k; 4,096 MB.
    self.assertIn('4.000 cpu-s/1k, 4,096 MB', md)


if __name__ == '__main__':
  absltest.main()
