// Entrypoint for gta6-twitch-ccu-forecast-01 (Node 22, standard library only).
// Reads every <id>.in case file from MAKEFUN_INPUT_DIR (or /input) in deterministic name order
// and writes validated 23-hour quantile forecasts to MAKEFUN_OUTPUT_DIR (or /output).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { forecastCase, type ForecastCaseInput, type ForecastHourOutput } from './lib/forecaster.ts';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MAX_OUTPUT_BYTES = 65_536;

function validateForecastRows(file: string, rows: readonly ForecastHourOutput[]): void {
  if (rows.length !== 23) {
    throw new Error(`${file}: expected 23 forecast rows, got ${rows.length}`);
  }
  const seen = new Set<number>();
  for (const row of rows) {
    if (!Number.isInteger(row.hour) || row.hour < 2 || row.hour > 24 || seen.has(row.hour)) {
      throw new Error(`${file}: invalid or duplicate hour ${row.hour}`);
    }
    seen.add(row.hour);
    if (!Array.isArray(row.q) || row.q.length !== 5) {
      throw new Error(`${file}: hour ${row.hour} must carry exactly 5 quantiles`);
    }
    for (let i = 0; i < 5; i++) {
      const v = row.q[i];
      if (!Number.isFinite(v) || v <= 0 || v > 1_000_000_000) {
        throw new Error(`${file}: hour ${row.hour} quantile[${i}] out of bounds: ${v}`);
      }
      if (i > 0 && v < row.q[i - 1]) {
        throw new Error(`${file}: hour ${row.hour} quantiles not non-decreasing`);
      }
    }
  }
}

function run(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const files = readdirSync(INPUT_DIR)
    .filter((f) => f.endsWith('.in'))
    .sort();

  for (const file of files) {
    const id = file.slice(0, -'.in'.length);
    const raw = readFileSync(join(INPUT_DIR, file), 'utf8');
    const parsed = JSON.parse(raw) as ForecastCaseInput;
    const hours = forecastCase(parsed)
      .slice()
      .sort((a, b) => a.hour - b.hour);
    validateForecastRows(file, hours);
    const payload = `${JSON.stringify({ hours })}\n`;
    if (Buffer.byteLength(payload, 'utf8') > MAX_OUTPUT_BYTES) {
      throw new Error(`${file}: serialized forecast exceeds ${MAX_OUTPUT_BYTES} bytes`);
    }
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), payload, 'utf8');
  }
}

run();
