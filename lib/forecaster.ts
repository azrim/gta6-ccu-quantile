// Kernel-weighted empirical log-ratio quantile forecaster for Twitch launch-day CCU.
// Pure Node 22 standard-library TypeScript: deterministic, offline, zero external dependencies.

export interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number | null;
  reads?: number;
}

export interface RawPoint {
  at: string;
  viewers: number;
  channels?: number | null;
}

export interface KnownHour {
  hour: number;
  from?: string;
  until?: string;
  viewers: number;
  channels?: number | null;
  reads?: number;
}

export interface ForecastSlot {
  hour: number;
  from?: string;
  until?: string;
}

export interface ComparableEntry {
  name: string;
  day_start: string;
  unlock: string;
  platforms: string;
  source?: string;
  hourly: HourlyPoint[];
}

export interface ForecastCaseInput {
  id: string;
  game: { name: string; twitch_id?: string };
  day_start: string;
  checkpoint?: string;
  levels: number[];
  known: KnownHour[];
  forecast: ForecastSlot[];
  history?: RawPoint[] | null;
  history_hourly?: HourlyPoint[] | null;
  notes?: string;
  comparables: ComparableEntry[];
}

export interface ForecastHourOutput {
  hour: number;
  q: number[];
}

export const TARGET_LEVELS: readonly number[] = [0.05, 0.25, 0.5, 0.75, 0.95];
const NORMAL_Z: readonly number[] = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const HOUR_MS = 3_600_000;
const MIN_VIEWERS = 0.01;
const MAX_VIEWERS = 1_000_000_000;

export interface LaunchProfile {
  name: string;
  dayStartIso: string;
  usDst: boolean;
  v1: number;
  logV1: number;
  mom1: number;
  mom6: number;
  dip: number;
  intraSurprise: number;
  nzLog: number;
  eaScore: number;
  logVpc: number;
  hasVpc: boolean;
  consoleOnly: boolean;
  isRolling: boolean;
  isSimulAfter: boolean;
  isPcMidnightUtc: boolean;
  isPcLaterDay: boolean;
  isF2P: boolean;
  actualViewers: number[]; // length 23 for hours 2..24 (empty on target)
  logRatios: number[];     // length 23: ln(V_h / V_1) for hours 2..24
}

export interface HourBaseStats {
  empQs: number[];
  center: number;
  baseSigma: number;
}

export interface CalibrationParams {
  empWeight: number;
  sigmaLow: number;
  sigmaHigh: number;
  tailBoost: number;
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

/** Deterministic US Daylight Saving Time check (2nd Sunday in March to 1st Sunday in November). */
export function isUsDst(iso: string): boolean {
  const d = new Date(iso);
  const y = d.getUTCFullYear();
  const m = d.getUTCMonth() + 1;
  const day = d.getUTCDate();
  if (m > 3 && m < 11) return true;
  if (m < 3 || m > 11) return false;
  if (m === 3) {
    const march1Dow = new Date(Date.UTC(y, 2, 1)).getUTCDay();
    const secondSunday = 8 + ((7 - march1Dow) % 7);
    return day > secondSunday;
  }
  const nov1Dow = new Date(Date.UTC(y, 10, 1)).getUTCDay();
  const firstSunday = 1 + ((7 - nov1Dow) % 7);
  return day < firstSunday;
}

function buildOffsetMap(
  dayStartMs: number,
  hourly: readonly HourlyPoint[] | null | undefined,
  rawHistory?: readonly RawPoint[] | null,
): Map<number, { viewers: number; channels: number | null }> {
  const byOff = new Map<number, { viewers: number; channels: number | null }>();

  if (rawHistory && rawHistory.length > 0) {
    const buckets = new Map<number, { vSum: number; cSum: number; cCnt: number; n: number }>();
    for (const pt of rawHistory) {
      const t = Date.parse(pt.at);
      if (!Number.isFinite(t) || !(pt.viewers >= 0)) continue;
      const off = Math.floor((t - dayStartMs) / HOUR_MS);
      if (off >= 0) continue;
      let b = buckets.get(off);
      if (!b) {
        b = { vSum: 0, cSum: 0, cCnt: 0, n: 0 };
        buckets.set(off, b);
      }
      b.vSum += pt.viewers;
      b.n += 1;
      if (pt.channels !== null && pt.channels !== undefined && pt.channels >= 0) {
        b.cSum += pt.channels;
        b.cCnt += 1;
      }
    }
    for (const [off, b] of buckets.entries()) {
      if (b.n > 0) {
        byOff.set(off, {
          viewers: b.vSum / b.n,
          channels: b.cCnt > 0 ? b.cSum / b.cCnt : null,
        });
      }
    }
  }

  if (hourly && hourly.length > 0) {
    for (const row of hourly) {
      const t = Date.parse(row.from);
      if (!Number.isFinite(t) || !(row.viewers >= 0)) continue;
      const off = Math.round((t - dayStartMs) / HOUR_MS);
      byOff.set(off, {
        viewers: row.viewers,
        channels: row.channels !== null && row.channels !== undefined ? row.channels : null,
      });
    }
  }

  return byOff;
}

function computeIntraHourSurprise(
  dayStartMs: number,
  v1: number,
  mom1: number,
  rawHistory?: readonly RawPoint[] | null,
): number {
  if (!rawHistory || rawHistory.length < 15) return 0;
  let totalInH1 = 0;
  let lateCount = 0;
  let lateSum = 0;
  const h1End = dayStartMs + HOUR_MS;
  const lateStart = dayStartMs + 45 * 60_000;
  for (const pt of rawHistory) {
    const t = Date.parse(pt.at);
    if (!Number.isFinite(t) || !(pt.viewers >= 0)) continue;
    if (t >= dayStartMs && t < h1End) {
      totalInH1 += 1;
      if (t >= lateStart) {
        lateCount += 1;
        lateSum += pt.viewers;
      }
    }
  }
  if (totalInH1 < 15 || lateCount < 5) return 0;
  const lateMean = Math.max(1, lateSum / lateCount);
  const rawDep = Math.log(lateMean / Math.max(1, v1));
  return clamp(rawDep - 0.38 * mom1, -0.25, 0.25);
}

function parseRegimeAndFeatures(
  name: string,
  dayStartIso: string,
  dayStartMs: number,
  v1Raw: number,
  ch1Raw: number | null,
  byOff: Map<number, { viewers: number; channels: number | null }>,
  rawHistory: readonly RawPoint[] | null | undefined,
  platformText: string,
  unlockText: string,
  actualViewers: number[],
  logRatios: number[],
): LaunchProfile {
  const v1 = Math.max(1, v1Raw);
  const logV1 = Math.log(v1);

  const v0 = byOff.get(-1)?.viewers ?? v1;
  const vm6 = byOff.get(-6)?.viewers ?? byOff.get(-5)?.viewers ?? v0;
  const mom1 = clamp(Math.log(v1 / Math.max(1, v0)), -2.5, 2.5);
  const intraSurprise = computeIntraHourSurprise(dayStartMs, v1, mom1, rawHistory);

  let prePrime = v0;
  for (let off = -9; off <= -1; off++) {
    const v = byOff.get(off)?.viewers;
    if (v !== undefined && v > prePrime) prePrime = v;
  }

  let pre42To16 = 0;
  for (let off = -42; off <= -16; off++) {
    const v = byOff.get(off)?.viewers;
    if (v !== undefined && v > pre42To16) pre42To16 = v;
  }

  let pre24To15 = 0;
  for (let off = -24; off <= -15; off++) {
    const v = byOff.get(off)?.viewers;
    if (v !== undefined && v > pre24To15) pre24To15 = v;
  }

  const nzPeak = Math.max(
    byOff.get(-13)?.viewers ?? 0,
    byOff.get(-12)?.viewers ?? 0,
    byOff.get(-11)?.viewers ?? 0,
  );
  const nzLift = (nzPeak + 50) / (pre24To15 + 50);
  const nzLog = clamp(Math.log(nzLift), -1.5, 4.5);

  const pLow = platformText.toLowerCase();
  const uLow = unlockText.toLowerCase();
  const fullLow = `${pLow} ${uLow}`;

  const explicitConsoleOnly =
    /console[- ]only/.test(fullLow) ||
    /\b(ps5|ps4|xbox|switch)\s+only\b/.test(fullLow);
  const cleanedPlat = pLow
    .replace(/\([^)]*later[^)]*\)/g, '')
    .replace(/\([^)]*pc\s+20\d\d[^)]*\)/g, '');
  const mentionsPc = /\b(pc|steam|stadia)\b/.test(cleanedPlat);
  const consoleOnly = explicitConsoleOnly || (!mentionsPc && /\b(ps5|ps4|playstation|xbox|switch)\b/.test(fullLow));

  const explicitNoEa = /no early access/.test(uLow);
  const explicitEa =
    !explicitNoEa &&
    (/early-access-contaminated/.test(uLow) ||
      ((/early[- ]access/.test(uLow)) && !/campaign early access/.test(uLow)));
  const empiricalEa = !explicitNoEa && v1 >= 5000 && pre42To16 / v1 > 0.85 && nzLift < 2.5;
  const eaScore = explicitEa ? 1.0 : empiricalEa ? 0.85 : 0.0;

  const isPcMidnightUtc =
    /23:00z|00:00z|midnight gmt/.test(uLow) ||
    (!consoleOnly && v1 > 100_000 && v0 / Math.max(1, vm6) > 4.0);
  const isPcLaterDay = /05:00z|pc unlock later|18:00z/.test(uLow);
  const isSimulAfter =
    v1 < 5500 ||
    ((/single.*global unlock|simultaneous/.test(uLow)) &&
      !/23:00z|00:00z|midnight gmt|north america simultaneous|no simultaneous/.test(uLow));
  const isRolling =
    ((/midnight local|rolling/.test(uLow)) && eaScore < 0.5) ||
    (v1 >= 10_000 && nzLift > 3.5 && eaScore < 0.5 && !isPcMidnightUtc);
  const isF2P = /free-to-play|\bf2p\b/.test(fullLow);

  const hasVpc = ch1Raw !== null && ch1Raw !== undefined && ch1Raw > 0;
  const logVpc = hasVpc ? clamp(Math.log(v1 / ch1Raw), 1.5, 6.5) : 3.85;

  return {
    name,
    dayStartIso,
    usDst: isUsDst(dayStartIso),
    v1,
    logV1,
    mom1,
    mom6: clamp(Math.log(v1 / Math.max(1, vm6)), -3.0, 3.0),
    dip: clamp(Math.log(Math.max(1, prePrime) / v1), -1.5, 2.5),
    intraSurprise,
    nzLog,
    eaScore,
    logVpc,
    hasVpc,
    consoleOnly,
    isRolling,
    isSimulAfter,
    isPcMidnightUtc,
    isPcLaterDay,
    isF2P,
    actualViewers,
    logRatios,
  };
}

export function parseValidComparables(comparables: readonly ComparableEntry[]): LaunchProfile[] {
  const result: LaunchProfile[] = [];
  for (const comp of comparables) {
    if (/target-history|not a launch/i.test(comp.unlock)) continue;
    const dayStartMs = Date.parse(comp.day_start);
    if (!Number.isFinite(dayStartMs)) continue;

    const byOff = buildOffsetMap(dayStartMs, comp.hourly, null);
    const h1 = byOff.get(0);
    if (!h1 || !(h1.viewers > 0)) continue;

    const actualViewers: number[] = [];
    const logRatios: number[] = [];
    let complete = true;
    for (let h = 2; h <= 24; h++) {
      const row = byOff.get(h - 1);
      if (!row || !(row.viewers > 0)) {
        complete = false;
        break;
      }
      actualViewers.push(row.viewers);
      logRatios.push(Math.log(row.viewers / h1.viewers));
    }
    if (!complete) continue;

    result.push(
      parseRegimeAndFeatures(
        comp.name,
        comp.day_start,
        dayStartMs,
        h1.viewers,
        h1.channels,
        byOff,
        null,
        comp.platforms ?? '',
        comp.unlock ?? '',
        actualViewers,
        logRatios,
      ),
    );
  }
  return result;
}

export function buildTargetProfile(caseData: ForecastCaseInput): LaunchProfile {
  const firstKnown = caseData.known.find((k) => k.hour === 1);
  if (!firstKnown) {
    throw new Error(`Case ${caseData.id} is missing hour-1 print in known`);
  }
  const dayStartMs = Date.parse(caseData.day_start);
  const byOff = buildOffsetMap(dayStartMs, caseData.history_hourly, caseData.history);

  // Merge pre-launch category history if present in comparables (e.g. target-history entry)
  let extraPlat = '';
  let extraUnlock = '';
  const gameNameLow = (caseData.game?.name ?? '').toLowerCase();
  for (const comp of caseData.comparables) {
    if (/target-history|not a launch/i.test(comp.unlock)) {
      if (gameNameLow && comp.name.toLowerCase().startsWith(gameNameLow)) {
        extraPlat = comp.platforms ?? '';
        extraUnlock = comp.unlock ?? '';
        for (const row of comp.hourly) {
          const t = Date.parse(row.from);
          if (!Number.isFinite(t) || !(row.viewers >= 0)) continue;
          const off = Math.round((t - dayStartMs) / HOUR_MS);
          if (off < 0 && !byOff.has(off)) {
            byOff.set(off, {
              viewers: row.viewers,
              channels: row.channels !== null && row.channels !== undefined ? row.channels : null,
            });
          }
        }
      }
    }
  }

  const notes = caseData.notes ?? '';
  const platMatch = notes.match(/released\s+\d{4}-\d{2}-\d{2}\s+on\s+([^;]+)/i);
  const platformText = `${platMatch ? platMatch[1] : ''} ${extraPlat}`.trim();
  const unlockText = `${notes} ${extraUnlock}`.trim();

  return parseRegimeAndFeatures(
    caseData.game?.name ?? caseData.id,
    caseData.day_start,
    dayStartMs,
    firstKnown.viewers,
    firstKnown.channels ?? null,
    byOff,
    caseData.history,
    platformText,
    unlockText,
    [],
    [],
  );
}

function weightedQuantile(values: readonly number[], weights: readonly number[], tau: number): number {
  const pairs: Array<{ v: number; w: number }> = [];
  let totalW = 0;
  for (let i = 0; i < values.length; i++) {
    const w = weights[i];
    if (w > 0 && Number.isFinite(values[i])) {
      pairs.push({ v: values[i], w });
      totalW += w;
    }
  }
  if (pairs.length === 0 || totalW <= 0) return 0;
  pairs.sort((a, b) => a.v - b.v);

  const mids: number[] = [];
  let cum = 0;
  for (let i = 0; i < pairs.length; i++) {
    mids.push((cum + 0.5 * pairs[i].w) / totalW);
    cum += pairs[i].w;
  }

  if (tau <= mids[0]) return pairs[0].v;
  const last = pairs.length - 1;
  if (tau >= mids[last]) return pairs[last].v;

  for (let i = 0; i < last; i++) {
    if (tau >= mids[i] && tau <= mids[i + 1]) {
      const span = Math.max(1e-12, mids[i + 1] - mids[i]);
      const frac = (tau - mids[i]) / span;
      return pairs[i].v + frac * (pairs[i + 1].v - pairs[i].v);
    }
  }
  return pairs[last].v;
}

export function computeKernelWeights(target: LaunchProfile, pool: readonly LaunchProfile[]): number[] {
  const tgtUnlaunched = target.v1 < 5500;
  const megaLaunch = !tgtUnlaunched && target.v1 > 200_000;
  const bwScale = megaLaunch ? 1.75 : 1.40;
  const penPlat = megaLaunch ? 1.05 : 1.45;
  const penPcMid = megaLaunch ? 0.85 : 1.35;

  const rawSharp = pool.map((comp) => {
    let d2 = 0;

    // 1. Live vs unlaunched-at-01Z regime separation
    const cmpUnlaunched = comp.v1 < 5500;
    d2 += Math.pow(((tgtUnlaunched ? 1 : 0) - (cmpUnlaunched ? 1 : 0)) * 4.5, 2);

    // 2. Scale proximity on log(V_1)
    d2 += Math.pow((target.logV1 - comp.logV1) / bwScale, 2);

    // 3. Pre-checkpoint evening prime-time dip & immediate transition momentum
    d2 += Math.pow((target.dip - comp.dip) / 0.90, 2);
    d2 += Math.pow((target.mom1 - comp.mom1) / 0.85, 2);
    d2 += Math.pow((target.nzLog - comp.nzLog) / 2.2, 2);

    // 4. Platform & unlock regime compatibility
    d2 += Math.pow((target.eaScore - comp.eaScore) * 1.85, 2);
    d2 += Math.pow(((target.consoleOnly ? 1 : 0) - (comp.consoleOnly ? 1 : 0)) * penPlat, 2);
    d2 += Math.pow(((target.isRolling ? 1 : 0) - (comp.isRolling ? 1 : 0)) * 1.35, 2);

    if (target.consoleOnly && comp.isPcLaterDay) {
      d2 += 1.75 * 1.75;
    }
    if (target.consoleOnly && comp.isPcMidnightUtc) {
      d2 += penPcMid * penPcMid;
    }
    if (!target.isSimulAfter && comp.isSimulAfter) {
      d2 += 2.4 * 2.4;
    }
    if (!target.isF2P && comp.isF2P) {
      d2 += 1.5 * 1.5;
    }

    // 5. Viewers-per-channel density when observable on both
    if (target.hasVpc && comp.hasVpc) {
      d2 += Math.pow((target.logVpc - comp.logVpc) / 1.6, 2);
    }

    return Math.exp(-0.5 * d2) + 1e-6;
  });

  // Broad regime diurnal prior weights to prevent sparse neighborhood collapse
  const rawPrior = pool.map((comp) => {
    const cmpUnlaunched = comp.v1 < 5500;
    let d2 = Math.pow(((tgtUnlaunched ? 1 : 0) - (cmpUnlaunched ? 1 : 0)) * 4.0, 2);
    d2 += Math.pow((target.logV1 - comp.logV1) / 2.8, 2);
    d2 += Math.pow((target.eaScore - comp.eaScore) * 1.1, 2);
    if (!target.isSimulAfter && comp.isSimulAfter) d2 += 2.0 * 2.0;
    return Math.exp(-0.5 * d2) + 1e-6;
  });

  const sharpSum = rawSharp.reduce((a, b) => a + b, 0);
  const priorSum = rawPrior.reduce((a, b) => a + b, 0);
  let sharpSq = 0;
  for (const w of rawSharp) sharpSq += (w / sharpSum) * (w / sharpSum);
  const nEff = sharpSq > 0 ? 1 / sharpSq : 1;
  const priorMix = clamp(0.85 / (nEff + 1.5), 0.06, 0.28);

  return rawSharp.map((w, idx) => (1 - priorMix) * (w / sharpSum) + priorMix * (rawPrior[idx] / priorSum));
}

function alignedComparableRatio(target: LaunchProfile, comp: LaunchProfile, hi: number): number {
  let baseR = comp.logRatios[hi];
  // Align US Eastern Midnight unlock hour (04:00Z EDT vs 05:00Z EST) across rolling launches
  if (target.isRolling && comp.isRolling && !comp.isPcLaterDay && hi >= 2 && hi <= 7) {
    if (target.usDst && !comp.usDst && hi + 1 < 23) {
      baseR = 0.75 * comp.logRatios[hi + 1] + 0.25 * comp.logRatios[hi];
    } else if (!target.usDst && comp.usDst && hi - 1 >= 0) {
      baseR = 0.75 * comp.logRatios[hi - 1] + 0.25 * comp.logRatios[hi];
    }
  }
  return baseR;
}

export function pinballLossLog(actualLogRatio: number, predLogRatio: number, tau: number): number {
  const diff = actualLogRatio - predLogRatio;
  return 2 * Math.max(tau * diff, (tau - 1) * diff);
}

export function computeHourBaseStats(
  target: LaunchProfile,
  pool: readonly LaunchProfile[],
): HourBaseStats[] {
  const weights = computeKernelWeights(target, pool);
  let wSum = 0;
  for (const w of weights) wSum += w;
  let sqSum = 0;
  for (const w of weights) {
    const p = w / wSum;
    sqSum += p * p;
  }
  const nEff = 1 / Math.max(1e-6, sqSum);
  const dofFactor = Math.sqrt((nEff + 0.8) / Math.max(2.0, nEff - 0.6));

  const tgtUnlaunched = target.v1 < 5500;
  const maxRefLogV1 = Math.log(850_000);
  const effTargetScale = Math.min(target.logV1, maxRefLogV1);
  const stats: HourBaseStats[] = [];

  for (let hi = 0; hi < 23; hi++) {
    const rampPrime = hi <= 1 ? 0 : Math.min(1, (hi - 1) / 12);
    const rampMom = Math.max(0, 1 - hi / 4);
    const rampDip = hi <= 2 ? 0 : Math.min(1, (hi - 2) / 11);
    const intraNudge = hi === 0 ? 0.50 * target.intraSurprise : hi === 1 ? 0.22 * target.intraSurprise : 0;

    const adjusted = pool.map((comp) => {
      const baseR = alignedComparableRatio(target, comp, hi);
      if (tgtUnlaunched) {
        const unlRamp = hi <= 3 ? 0.15 : Math.min(1, (hi - 2) / 10);
        return baseR - 0.72 * unlRamp * (target.logV1 - comp.logV1) + intraNudge;
      }
      const effCompScale = Math.min(comp.logV1, maxRefLogV1);
      const scaleShift = -0.24 * rampPrime * (effTargetScale - effCompScale);
      const dipShift = 0.22 * rampDip * (target.dip - comp.dip);
      const momShift = 0.22 * rampMom * (target.mom1 - comp.mom1);
      return baseR + scaleShift + dipShift + momShift + intraNudge;
    });

    let mean = 0;
    for (let i = 0; i < adjusted.length; i++) {
      mean += (weights[i] / wSum) * adjusted[i];
    }
    const empQs = TARGET_LEVELS.map((tau) => weightedQuantile(adjusted, weights, tau));
    const center = 0.58 * empQs[2] + 0.42 * mean;

    let varW = 0;
    for (let i = 0; i < adjusted.length; i++) {
      const diff = adjusted[i] - center;
      varW += (weights[i] / wSum) * diff * diff;
    }
    const std = Math.sqrt(Math.max(0.008, varW));
    const iqrStd = Math.max(0.07, (empQs[3] - empQs[1]) / 1.3489795);
    const baseSigma = dofFactor * (0.55 * std + 0.45 * iqrStd);

    stats.push({ empQs, center, baseSigma });
  }

  return stats;
}

export function synthesizeQuantileRows(
  stats: readonly HourBaseStats[],
  params: CalibrationParams,
): number[][] {
  const raw = stats.map(({ empQs, center, baseSigma }) => {
    const row = TARGET_LEVELS.map((_, ti) => {
      const asym = ti < 2 ? params.sigmaLow : ti > 2 ? params.sigmaHigh : 1.0;
      const tail = ti === 0 || ti === 4 ? params.tailBoost : 1.0;
      const parQ = center + NORMAL_Z[ti] * baseSigma * asym * tail;
      const ew = ti === 0 || ti === 4 ? 0.5 * params.empWeight : params.empWeight;
      return ew * empQs[ti] + (1 - ew) * parQ;
    });
    for (let ti = 1; ti < row.length; ti++) {
      if (row[ti] < row[ti - 1]) row[ti] = row[ti - 1];
    }
    return row;
  });

  return raw.map((row, hi) => {
    // Preserve sharp midnight/US-unlock step dynamics in hours 2..6 (hi < 5) and endpoint hour 24
    if (hi < 5 || hi === 22) return row;
    const smoothed = row.map((val, ti) => 0.15 * raw[hi - 1][ti] + 0.70 * val + 0.15 * raw[hi + 1][ti]);
    for (let ti = 1; ti < smoothed.length; ti++) {
      if (smoothed[ti] < smoothed[ti - 1]) smoothed[ti] = smoothed[ti - 1];
    }
    return smoothed;
  });
}

export function calibrateSpreadLoo(
  target: LaunchProfile,
  pool: readonly LaunchProfile[],
): CalibrationParams {
  const fallback: CalibrationParams = {
    empWeight: 0.35,
    sigmaLow: 0.96,
    sigmaHigh: 1.10,
    tailBoost: 1.10,
  };
  if (pool.length < 6) return fallback;

  const relWeights = computeKernelWeights(target, pool);
  let relSum = 0;
  for (const w of relWeights) relSum += w;

  const active: Array<{ comp: LaunchProfile; weight: number; stats: HourBaseStats[] }> = [];
  let activeWeightSum = 0;
  for (let k = 0; k < pool.length; k++) {
    const normW = relWeights[k] / relSum;
    if (normW >= 0.01) {
      const subPool = pool.filter((_, m) => m !== k);
      active.push({
        comp: pool[k],
        weight: normW,
        stats: computeHourBaseStats(pool[k], subPool),
      });
      activeWeightSum += normW;
    }
  }
  if (active.length < 3 || activeWeightSum <= 0) return fallback;

  const empGrid = [0.25, 0.38];
  const loGrid = [0.90, 1.00];
  const hiGrid = [1.02, 1.14, 1.26];
  const tailGrid = [1.04, 1.14];

  let bestScore = Infinity;
  let best = fallback;

  for (const empWeight of empGrid) {
    for (const sigmaLow of loGrid) {
      for (const sigmaHigh of hiGrid) {
        for (const tailBoost of tailGrid) {
          const cand: CalibrationParams = { empWeight, sigmaLow, sigmaHigh, tailBoost };
          let weightedLoss = 0;
          for (const item of active) {
            const rows = synthesizeQuantileRows(item.stats, cand);
            let itemLoss = 0;
            for (let hi = 0; hi < 23; hi++) {
              const rActual = item.comp.logRatios[hi];
              for (let ti = 0; ti < 5; ti++) {
                itemLoss += pinballLossLog(rActual, rows[hi][ti], TARGET_LEVELS[ti]);
              }
            }
            weightedLoss += (item.weight / activeWeightSum) * (itemLoss / 115);
          }
          const ridge =
            0.018 *
            (Math.pow(empWeight - 0.35, 2) +
              Math.pow(sigmaLow - 0.96, 2) +
              Math.pow(sigmaHigh - 1.10, 2) +
              Math.pow(tailBoost - 1.10, 2));
          const totalObj = weightedLoss + ridge;
          if (totalObj < bestScore) {
            bestScore = totalObj;
            best = cand;
          }
        }
      }
    }
  }

  return best;
}

export function predictLogRatioQuantiles(
  target: LaunchProfile,
  pool: readonly LaunchProfile[],
): number[][] {
  if (pool.length === 0) {
    const rows: number[][] = [];
    for (let hi = 0; hi < 23; hi++) {
      const h = hi + 2;
      rows.push(NORMAL_Z.map((z) => z * 0.15 * Math.sqrt(h - 1)));
    }
    return rows;
  }

  const baseStats = computeHourBaseStats(target, pool);
  const calibrated = calibrateSpreadLoo(target, pool);
  return synthesizeQuantileRows(baseStats, calibrated);
}

export function forecastCase(caseData: ForecastCaseInput): ForecastHourOutput[] {
  if (
    !Array.isArray(caseData.levels) ||
    caseData.levels.length !== TARGET_LEVELS.length ||
    caseData.levels.some((lv, idx) => lv !== TARGET_LEVELS[idx])
  ) {
    throw new Error(`Unexpected quantile levels in case ${caseData.id}`);
  }

  const target = buildTargetProfile(caseData);
  const pool = parseValidComparables(caseData.comparables ?? []);
  const logQuantilesByHour = predictLogRatioQuantiles(target, pool);

  return caseData.forecast.map(({ hour }) => {
    const hi = hour - 2;
    if (hi < 0 || hi >= 23) {
      throw new Error(`Unexpected forecast hour ${hour} in case ${caseData.id}`);
    }
    const logRow = logQuantilesByHour[hi];
    const q: number[] = [];
    for (let ti = 0; ti < TARGET_LEVELS.length; ti++) {
      const rawVal = target.v1 * Math.exp(logRow[ti]);
      const bounded = clamp(round2(rawVal), MIN_VIEWERS, MAX_VIEWERS);
      q.push(ti > 0 && bounded < q[ti - 1] ? q[ti - 1] : bounded);
    }
    return { hour, q };
  });
}
