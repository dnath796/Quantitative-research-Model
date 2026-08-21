"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type OptionType = "call" | "put";
type Style = "european" | "american";
type Side = "long" | "short";
type Inputs = {
  spot: number; strike: number; maturity: number; ratePct: number;
  volatilityPct: number; dividendPct: number; optionType: OptionType;
  style: Style; position: Side; quantity: number; multiplier: number;
  premium: number; premiumMode: "model" | "market"; simulations: number;
  realizedVolPct: number; hedgeSteps: number; hedgePaths: number; transactionCostBps: number;
};

const DEFAULTS: Inputs = {
  spot: 100, strike: 105, maturity: 0.5, ratePct: 5, volatilityPct: 25,
  dividendPct: 2, optionType: "call", style: "european", position: "long",
  quantity: 10, multiplier: 100, premium: 5.52, premiumMode: "model",
  simulations: 20_000, realizedVolPct: 25, hedgeSteps: 52, hedgePaths: 750, transactionCostBps: 1,
};

const normPdf = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
function normCdf(x: number) {
  const sign = x < 0 ? -1 : 1;
  const z = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * z);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return 0.5 * (1 + sign * erf);
}
function intrinsic(S: number, K: number, type: OptionType) {
  return type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
}
function blackScholes(S: number, K: number, T: number, r: number, sigma: number, q: number, type: OptionType) {
  if (T <= 0) return intrinsic(S, K, type);
  if (sigma <= 0) {
    const pv = S * Math.exp(-q * T) - K * Math.exp(-r * T);
    return type === "call" ? Math.max(pv, 0) : Math.max(-pv, 0);
  }
  const rootT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * rootT);
  const d2 = d1 - sigma * rootT;
  return type === "call"
    ? S * Math.exp(-q * T) * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2)
    : K * Math.exp(-r * T) * normCdf(-d2) - S * Math.exp(-q * T) * normCdf(-d1);
}
function binomial(S: number, K: number, T: number, r: number, sigma: number, q: number, type: OptionType, steps: number, american: boolean) {
  if (T <= 0 || sigma <= 0) return blackScholes(S, K, T, r, sigma, q, type);
  const dt = T / steps;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = 1 / u;
  const p = (Math.exp((r - q) * dt) - d) / (u - d);
  if (p < 0 || p > 1) return Number.NaN;
  let values = Array.from({ length: steps + 1 }, (_, j) => intrinsic(S * u ** (steps - j) * d ** j, K, type));
  const discount = Math.exp(-r * dt);
  for (let level = steps - 1; level >= 0; level -= 1) {
    const next = new Array<number>(level + 1);
    for (let j = 0; j <= level; j += 1) {
      const continuation = discount * (p * values[j] + (1 - p) * values[j + 1]);
      const exercise = intrinsic(S * u ** (level - j) * d ** j, K, type);
      next[j] = american ? Math.max(continuation, exercise) : continuation;
    }
    values = next;
  }
  return values[0];
}
function greeks(S: number, K: number, T: number, r: number, sigma: number, q: number, type: OptionType) {
  if (T <= 0 || sigma <= 0) return { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
  const rootT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * rootT);
  const d2 = d1 - sigma * rootT;
  const sd = Math.exp(-q * T);
  const kd = Math.exp(-r * T);
  const commonTheta = -(S * sd * normPdf(d1) * sigma) / (2 * rootT);
  if (type === "call") return {
    delta: sd * normCdf(d1), gamma: sd * normPdf(d1) / (S * sigma * rootT),
    theta: commonTheta - r * K * kd * normCdf(d2) + q * S * sd * normCdf(d1),
    vega: S * sd * normPdf(d1) * rootT, rho: K * T * kd * normCdf(d2),
  };
  return {
    delta: sd * (normCdf(d1) - 1), gamma: sd * normPdf(d1) / (S * sigma * rootT),
    theta: commonTheta + r * K * kd * normCdf(-d2) - q * S * sd * normCdf(-d1),
    vega: S * sd * normPdf(d1) * rootT, rho: -K * T * kd * normCdf(-d2),
  };
}
function impliedVolatility(price: number, S: number, K: number, T: number, r: number, q: number, type: OptionType) {
  if (price <= 0 || T <= 0) return null;
  const ds = S * Math.exp(-q * T);
  const dk = K * Math.exp(-r * T);
  const lower = type === "call" ? Math.max(0, ds - dk) : Math.max(0, dk - ds);
  const upper = type === "call" ? ds : dk;
  if (price < lower || price >= upper) return null;
  let low = 0.000001, high = 5;
  for (let i = 0; i < 100; i += 1) {
    const mid = (low + high) / 2;
    if (blackScholes(S, K, T, r, mid, q, type) > price) high = mid; else low = mid;
  }
  return (low + high) / 2;
}

function seededRandom(seed = 42) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function normalDraw(random: () => number) {
  const u1 = Math.max(random(), 1e-12), u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function monteCarlo(S: number, K: number, T: number, r: number, sigma: number, q: number, type: OptionType, simulations: number) {
  if (T <= 0) return { price: intrinsic(S, K, type), se: 0, low: intrinsic(S, K, type), high: intrinsic(S, K, type) };
  const requested = Number.isFinite(simulations) ? simulations : DEFAULTS.simulations;
  const count = Math.max(100, Math.min(Math.round(requested), 200_000));
  const random = seededRandom(42);
  let sum = 0, sumSquares = 0, generated = 0;
  const drift = (r - q - 0.5 * sigma * sigma) * T, diffusion = sigma * Math.sqrt(T), discount = Math.exp(-r * T);
  while (generated < count) {
    const z = normalDraw(random);
    for (const draw of [z, -z]) {
      if (generated >= count) break;
      const terminal = S * Math.exp(drift + diffusion * draw);
      const value = discount * intrinsic(terminal, K, type);
      sum += value; sumSquares += value * value; generated += 1;
    }
  }
  const price = sum / count;
  const variance = Math.max((sumSquares - count * price * price) / (count - 1), 0);
  const se = Math.sqrt(variance / count);
  return { price, se, low: price - 1.96 * se, high: price + 1.96 * se };
}

function simulateHedging(S: number, K: number, T: number, r: number, impliedVol: number, realizedVol: number, q: number, type: OptionType, stepsInput: number, pathsInput: number, costsBps: number, multiplier: number) {
  const requestedSteps = Number.isFinite(stepsInput) ? stepsInput : DEFAULTS.hedgeSteps;
  const requestedPaths = Number.isFinite(pathsInput) ? pathsInput : DEFAULTS.hedgePaths;
  const steps = Math.max(1, Math.min(Math.round(requestedSteps), 252));
  const paths = Math.max(100, Math.min(Math.round(requestedPaths), 5_000));
  const dt = T / steps, random = seededRandom(314159), costRate = Math.max(costsBps, 0) / 10_000;
  const premium = blackScholes(S, K, T, r, impliedVol, q, type);
  const initialDelta = greeks(S, K, T, r, impliedVol, q, type).delta;
  const pnl = new Array<number>(paths);
  for (let path = 0; path < paths; path += 1) {
    let spot = S, shares = initialDelta, cash = premium - shares * S;
    for (let step = 1; step <= steps; step += 1) {
      const priorSpot = spot;
      spot *= Math.exp((r - q - 0.5 * realizedVol * realizedVol) * dt + realizedVol * Math.sqrt(dt) * normalDraw(random));
      cash = cash * Math.exp(r * dt) + shares * priorSpot * (Math.exp(q * dt) - 1);
      if (step < steps) {
        const nextDelta = greeks(spot, K, T - step * dt, r, impliedVol, q, type).delta;
        const trade = nextDelta - shares;
        cash -= trade * spot + Math.abs(trade) * spot * costRate;
        shares = nextDelta;
      }
    }
    pnl[path] = (cash + shares * spot - intrinsic(spot, K, type)) * multiplier;
  }
  pnl.sort((a, b) => a - b);
  const mean = pnl.reduce((a, b) => a + b, 0) / paths;
  const std = Math.sqrt(pnl.reduce((total, value) => total + (value - mean) ** 2, 0) / Math.max(paths - 1, 1));
  const quantile = (p: number) => pnl[Math.min(paths - 1, Math.floor((paths - 1) * p))];
  const bins = 17, min = pnl[0], max = pnl[paths - 1], width = Math.max((max - min) / bins, 1e-9);
  const histogram = Array.from({ length: bins }, (_, i) => ({ start: min + i * width, count: 0 }));
  pnl.forEach((value) => { histogram[Math.min(bins - 1, Math.floor((value - min) / width))].count += 1; });
  return { mean, std, p05: quantile(.05), median: quantile(.5), p95: quantile(.95), histogram, paths, steps };
}

function chainMoneyness(spot: number, strike: number, type: OptionType) {
  const relative = strike / spot - 1;
  if (Math.abs(relative) <= .025) return "ATM";
  return type === "call" ? (strike < spot ? "ITM" : "OTM") : (strike > spot ? "ITM" : "OTM");
}

function deterministicNoise(strike: number, days: number, type: OptionType, salt: number) {
  const value = Math.sin(strike * 12.9898 + days * 78.233 + (type === "call" ? 19.19 : 47.47) + salt * 31.17) * 43758.5453;
  return value - Math.floor(value);
}

function generateChain(S: number, r: number, sigma: number, q: number) {
  const expirations = [30, 60, 90, 180, 365];
  const step = Math.max(1, Math.round(S * .05 / 2.5) * 2.5);
  const center = Math.round(S / step) * step;
  const strikes = Array.from({ length: 13 }, (_, index) => Math.max(step, center + (index - 6) * step));
  return expirations.flatMap((days) => {
    const T = days / 365;
    return strikes.flatMap((strike) => (["call", "put"] as OptionType[]).map((type) => {
      const logMoneyness = Math.log(strike / S);
      const surfaceIv = Math.max(.0001, sigma - .12 * logMoneyness + .30 * logMoneyness ** 2 + .015 * Math.sqrt(T));
      const fair = blackScholes(S, strike, T, r, surfaceIv, q, type);
      const theoretical = blackScholes(S, strike, T, r, sigma, q, type);
      const halfSpread = Math.max(.01, .008 * fair + .015 + .12 * Math.abs(logMoneyness));
      const mid = Math.max(.005, fair + (deterministicNoise(strike, days, type, 1) - .5) * halfSpread * .24);
      const bid = Math.max(0, mid - halfSpread), ask = mid + halfSpread;
      const last = Math.max(0, mid + (deterministicNoise(strike, days, type, 2) - .5) * halfSpread * .7);
      const liquidity = Math.exp(-7.5 * Math.abs(logMoneyness)) * Math.exp(-.45 * T);
      const volume = Math.round(15 + 900 * liquidity * (.75 + .5 * deterministicNoise(strike, days, type, 3)));
      const openInterest = Math.max(volume, Math.round(80 + 4500 * liquidity * (.75 + .5 * deterministicNoise(strike, days, type, 4))));
      const risk = greeks(S, strike, T, r, surfaceIv, q, type);
      const intrinsicValue = intrinsic(S, strike, type);
      return { days, type, strike, moneyness: chainMoneyness(S, strike, type), bid, ask, mid, last, spread: ask - bid, volume, openInterest, iv: impliedVolatility(mid, S, strike, T, r, q, type) ?? surfaceIv, delta: risk.delta, gamma: risk.gamma, theta: risk.theta / 365, vega: risk.vega / 100, rho: risk.rho / 100, intrinsic: intrinsicValue, extrinsic: Math.max(mid - intrinsicValue, 0), theoretical, edge: mid - theoretical };
    }));
  });
}

const money = (value: number) => Number.isFinite(value) ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value) : "Unlimited";
const number = (value: number, digits = 4) => Number.isFinite(value) ? value.toLocaleString("en-US", { maximumFractionDigits: digits }) : "—";

function NumberField({ label, value, onChange, min, step = "any", suffix, hint }: { label: string; value: number; onChange: (value: number) => void; min?: number; step?: number | "any"; suffix?: string; hint?: string }) {
  return <label className="field"><span className="field-label">{label}</span><span className="input-shell"><input type="number" value={value} min={min} step={step} onChange={(e) => onChange(Number(e.target.value))} />{suffix && <span className="input-suffix">{suffix}</span>}</span>{hint && <small>{hint}</small>}</label>;
}
function Segmented<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (value: T) => void }) {
  return <fieldset className="field segment-field"><legend className="field-label">{label}</legend><div className="segmented">{options.map((option) => <button key={option.value} type="button" className={value === option.value ? "active" : ""} onClick={() => onChange(option.value)} aria-pressed={value === option.value}>{option.label}</button>)}</div></fieldset>;
}

function PayoffChart({ inputs, premium }: { inputs: Inputs; premium: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
      const ctx = canvas.getContext("2d"); if (!ctx) return;
      ctx.scale(ratio, ratio);
      const width = rect.width, height = rect.height;
      const pad = { left: 58, right: 20, top: 28, bottom: 42 };
      const minX = Math.max(0, Math.min(inputs.spot, inputs.strike) * 0.4);
      const maxX = Math.max(inputs.spot, inputs.strike) * 1.65;
      const sign = inputs.position === "long" ? 1 : -1;
      const units = inputs.quantity * inputs.multiplier;
      const points = Array.from({ length: 181 }, (_, i) => {
        const x = minX + (maxX - minX) * i / 180;
        const payoff = sign * units * intrinsic(x, inputs.strike, inputs.optionType);
        return { x, payoff, pnl: payoff - sign * units * premium };
      });
      const values = points.flatMap((p) => [p.payoff, p.pnl]);
      let minY = Math.min(...values, 0), maxY = Math.max(...values, 0);
      const extra = Math.max((maxY - minY) * 0.12, 1); minY -= extra; maxY += extra;
      const xs = (x: number) => pad.left + (x - minX) / (maxX - minX) * (width - pad.left - pad.right);
      const ys = (y: number) => pad.top + (maxY - y) / (maxY - minY) * (height - pad.top - pad.bottom);
      ctx.clearRect(0, 0, width, height); ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = "#73807c"; ctx.strokeStyle = "#dfe5e1"; ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i += 1) {
        const y = minY + (maxY - minY) * i / 4, py = ys(y);
        ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(width - pad.right, py); ctx.stroke();
        ctx.textAlign = "right"; ctx.fillText(Math.abs(y) >= 1000 ? `${(y / 1000).toFixed(1)}k` : y.toFixed(0), pad.left - 9, py + 4);
      }
      for (let i = 0; i <= 4; i += 1) { const x = minX + (maxX - minX) * i / 4; ctx.textAlign = "center"; ctx.fillText(`$${x.toFixed(0)}`, xs(x), height - 17); }
      ctx.strokeStyle = "#8e9994"; ctx.beginPath(); ctx.moveTo(pad.left, ys(0)); ctx.lineTo(width - pad.right, ys(0)); ctx.stroke();
      ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(xs(inputs.strike), pad.top); ctx.lineTo(xs(inputs.strike), height - pad.bottom); ctx.stroke(); ctx.setLineDash([]);
      const line = (key: "payoff" | "pnl", color: string) => { ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2.5; points.forEach((p, i) => i ? ctx.lineTo(xs(p.x), ys(p[key])) : ctx.moveTo(xs(p.x), ys(p[key]))); ctx.stroke(); };
      line("payoff", "#1f6f68"); line("pnl", "#e26345");
    };
    draw(); window.addEventListener("resize", draw); return () => window.removeEventListener("resize", draw);
  }, [inputs, premium]);
  return <canvas ref={canvasRef} className="payoff-canvas" role="img" aria-label="Expiration payoff and profit loss chart" />;
}

export default function Home() {
  const [inputs, setInputs] = useState<Inputs>(DEFAULTS);
  const [chainExpiration, setChainExpiration] = useState(30);
  const [chainType, setChainType] = useState<"all" | OptionType>("all");
  const [chainMoneynessFilter, setChainMoneynessFilter] = useState<"ALL" | "ITM" | "ATM" | "OTM">("ALL");
  const set = <K extends keyof Inputs>(key: K, value: Inputs[K]) => setInputs((current) => ({ ...current, [key]: value }));
  const analytics = useMemo(() => {
    const S = Math.max(inputs.spot, 0.01), K = Math.max(inputs.strike, 0.01), T = Math.max(inputs.maturity, 0);
    const r = inputs.ratePct / 100, sigma = Math.max(inputs.volatilityPct / 100, 0), q = inputs.dividendPct / 100;
    const bs = blackScholes(S, K, T, r, sigma, q, inputs.optionType);
    const premium = inputs.premiumMode === "model" ? bs : Math.max(inputs.premium, 0);
    const american = inputs.style === "american";
    const b100 = binomial(S, K, T, r, sigma, q, inputs.optionType, 100, american);
    const b500 = binomial(S, K, T, r, sigma, q, inputs.optionType, 500, american);
    const mc = monteCarlo(S, K, T, r, sigma, q, inputs.optionType, inputs.simulations);
    const unitGreeks = greeks(S, K, T, r, sigma, q, inputs.optionType);
    const sign = inputs.position === "long" ? 1 : -1;
    const units = Math.max(inputs.quantity, 0) * Math.max(inputs.multiplier, 0), factor = sign * units;
    const positionGreeks = Object.fromEntries(Object.entries(unitGreeks).map(([key, value]) => [key, value * factor])) as typeof unitGreeks;
    const breakEven = inputs.optionType === "call" ? K + premium : K - premium;
    let maxProfit = 0, maxLoss = 0;
    if (inputs.optionType === "call" && sign === 1) { maxProfit = Infinity; maxLoss = premium * units; }
    if (inputs.optionType === "call" && sign === -1) { maxProfit = premium * units; maxLoss = Infinity; }
    if (inputs.optionType === "put" && sign === 1) { maxProfit = Math.max(K - premium, 0) * units; maxLoss = premium * units; }
    if (inputs.optionType === "put" && sign === -1) { maxProfit = premium * units; maxLoss = Math.max(K - premium, 0) * units; }
    const iv = inputs.premiumMode === "market" ? impliedVolatility(premium, S, K, T, r, q, inputs.optionType) : sigma;
    const specs = [
      { name: "Base", spot: S, vol: sigma, rate: r, time: T }, { name: "Stock +10%", spot: S * 1.1, vol: sigma, rate: r, time: T },
      { name: "Stock −10%", spot: S * 0.9, vol: sigma, rate: r, time: T }, { name: "Vol +5 pts", spot: S, vol: sigma + 0.05, rate: r, time: T },
      { name: "Vol −5 pts", spot: S, vol: Math.max(sigma - 0.05, 0), rate: r, time: T }, { name: "30 days later", spot: S, vol: sigma, rate: r, time: Math.max(T - 30 / 365, 0) },
      { name: "Rates +1 pt", spot: S, vol: sigma, rate: r + 0.01, time: T },
    ];
    const scenarios = specs.map((s) => { const value = blackScholes(s.spot, K, s.time, s.rate, s.vol, q, inputs.optionType); return { ...s, value, pnl: (value - bs) * factor }; });
    const convergence = [5, 10, 25, 50, 100, 250, 500].map((steps) => ({ steps, value: binomial(S, K, T, r, sigma, q, inputs.optionType, steps, american) }));
    const mcConvergence = [1_000, 5_000, 10_000, 20_000, 50_000].map((simulations) => ({ simulations, ...monteCarlo(S, K, T, r, sigma, q, inputs.optionType, simulations) }));
    const hedge = simulateHedging(S, K, Math.max(T, 1 / 365), r, Math.max(sigma, .0001), Math.max(inputs.realizedVolPct / 100, 0), q, inputs.optionType, inputs.hedgeSteps, inputs.hedgePaths, inputs.transactionCostBps, Math.max(inputs.multiplier, 0));
    const surfaceStrikes = [.8, .9, 1, 1.1, 1.2].map((ratio) => S * ratio);
    const surfaceMaturities = [.1, .25, .5, 1, 2];
    const surface = surfaceMaturities.map((maturity) => ({ maturity, values: surfaceStrikes.map((strike) => {
      const logMoneyness = Math.log(strike / S);
      return Math.max(.0001, sigma - .12 * logMoneyness + .30 * logMoneyness ** 2 + .015 * Math.sqrt(maturity));
    }) }));
    const fullChain = generateChain(S, r, Math.max(sigma, .0001), q);
    return { bs, premium, b100, b500, mc, positionGreeks, breakEven, maxProfit, maxLoss, iv, scenarios, convergence, mcConvergence, hedge, surface, surfaceStrikes, fullChain, units };
  }, [inputs]);
  const chainRows = analytics.fullChain.filter((row) => row.days === chainExpiration && (chainType === "all" || row.type === chainType) && (chainMoneynessFilter === "ALL" || row.moneyness === chainMoneynessFilter));
  const optionLabel = `${inputs.style === "american" ? "American" : "European"} ${inputs.optionType === "call" ? "Call" : "Put"}`;

  return <main>
    <header className="topbar"><div className="brand"><span className="brand-mark">Δ</span><span>DerivativeLab</span></div><div className="live-pill"><span /> Live repricing</div></header>
    <section className="hero"><div><p className="eyebrow">DERIVATIVES PRICING &amp; RISK WORKBENCH</p><h1>From market inputs to<br /><em>decision-ready analytics.</em></h1><p className="hero-copy">Price with Black–Scholes, CRR trees, and Monte Carlo; map volatility, inspect Greeks, and simulate dynamic hedging—all in one live model.</p></div><div className="hero-formula" aria-label="Black Scholes formula"><span>European call</span><strong>C = Se<sup>−qT</sup>N(d₁) − Ke<sup>−rT</sup>N(d₂)</strong><small>Analytical · lattice · simulation</small></div></section>
    <section className="workspace">
      <aside className="input-panel">
        <div className="panel-heading"><div><p className="kicker">MODEL INPUTS</p><h2>Contract &amp; market</h2></div><button className="reset" type="button" onClick={() => setInputs(DEFAULTS)}>Reset</button></div>
        <Segmented label="Option type" value={inputs.optionType} options={[{ value: "call", label: "Call" }, { value: "put", label: "Put" }]} onChange={(v) => set("optionType", v)} />
        <Segmented label="Exercise style" value={inputs.style} options={[{ value: "european", label: "European" }, { value: "american", label: "American" }]} onChange={(v) => set("style", v)} />
        <div className="field-grid"><NumberField label="Spot price" value={inputs.spot} min={.01} step={1} suffix="$" onChange={(v) => set("spot", v)} /><NumberField label="Strike price" value={inputs.strike} min={.01} step={1} suffix="$" onChange={(v) => set("strike", v)} /><NumberField label="Time to expiry" value={inputs.maturity} min={0} step={.05} suffix="yr" onChange={(v) => set("maturity", v)} /><NumberField label="Volatility" value={inputs.volatilityPct} min={0} step={.5} suffix="%" onChange={(v) => set("volatilityPct", v)} /><NumberField label="Risk-free rate" value={inputs.ratePct} step={.25} suffix="%" onChange={(v) => set("ratePct", v)} /><NumberField label="Dividend yield" value={inputs.dividendPct} step={.25} suffix="%" onChange={(v) => set("dividendPct", v)} /></div>
        <div className="divider" /><Segmented label="Position" value={inputs.position} options={[{ value: "long", label: "Long" }, { value: "short", label: "Short" }]} onChange={(v) => set("position", v)} />
        <div className="field-grid"><NumberField label="Contracts" value={inputs.quantity} min={0} step={1} onChange={(v) => set("quantity", v)} /><NumberField label="Multiplier" value={inputs.multiplier} min={0} step={1} onChange={(v) => set("multiplier", v)} /></div>
        <Segmented label="Premium source" value={inputs.premiumMode} options={[{ value: "model", label: "Model price" }, { value: "market", label: "Market premium" }]} onChange={(v) => set("premiumMode", v)} />
        {inputs.premiumMode === "market" && <NumberField label="Observed premium" value={inputs.premium} min={0} step={.01} suffix="$" hint="Used for P&L and implied volatility" onChange={(v) => set("premium", v)} />}
        <details className="advanced-inputs"><summary>Simulation settings</summary><div className="field-grid"><NumberField label="Monte Carlo paths" value={inputs.simulations} min={100} step={1000} onChange={(v) => set("simulations", v)} /><NumberField label="Realized volatility" value={inputs.realizedVolPct} min={0} step={.5} suffix="%" onChange={(v) => set("realizedVolPct", v)} /><NumberField label="Hedge rebalances" value={inputs.hedgeSteps} min={1} step={1} onChange={(v) => set("hedgeSteps", v)} /><NumberField label="Hedge paths" value={inputs.hedgePaths} min={100} step={100} onChange={(v) => set("hedgePaths", v)} /><NumberField label="Trading cost" value={inputs.transactionCostBps} min={0} step={.5} suffix="bps" onChange={(v) => set("transactionCostBps", v)} /></div></details>
        <div className="input-summary"><span>{optionLabel}</span><strong>{inputs.position.toUpperCase()} · {number(analytics.units, 0)} units</strong></div>
      </aside>
      <div className="results">
        <div className="section-title"><div><p className="kicker">LIVE OUTPUT</p><h2>Pricing overview</h2></div><span className="as-of">Inputs update instantly</span></div>
        <div className="price-grid"><article className="price-card primary"><p>Black–Scholes</p><strong>{money(analytics.bs)}</strong><span>European benchmark · per unit</span></article><article className="price-card"><p>CRR binomial · 100</p><strong>{money(analytics.b100)}</strong><span>{inputs.style} exercise</span></article><article className="price-card"><p>CRR binomial · 500</p><strong>{money(analytics.b500)}</strong><span>{inputs.style} exercise</span></article><article className="price-card"><p>Monte Carlo</p><strong>{money(analytics.mc.price)}</strong><span>± {money(1.96 * analytics.mc.se)} at 95%</span></article><article className="price-card accent"><p>Implied volatility</p><strong>{analytics.iv === null ? "Outside bounds" : `${number(analytics.iv * 100, 2)}%`}</strong><span>{inputs.premiumMode === "market" ? `From ${money(analytics.premium)} premium` : "Matches model input"}</span></article></div>
        <div className="analytics-grid">
          <article className="card chart-card"><div className="card-title"><div><p className="kicker">EXPIRATION PROFILE</p><h3>Payoff &amp; profit / loss</h3></div><div className="legend"><span className="teal" />Payoff <span className="coral" />P&amp;L</div></div><PayoffChart inputs={inputs} premium={analytics.premium} /><div className="risk-strip"><div><span>Break-even</span><strong>{money(analytics.breakEven)}</strong></div><div><span>Maximum profit</span><strong>{money(analytics.maxProfit)}</strong></div><div><span>Maximum loss</span><strong>{money(analytics.maxLoss)}</strong></div></div></article>
          <article className="card greeks-card"><div className="card-title"><div><p className="kicker">POSITION RISK</p><h3>Portfolio Greeks</h3></div><span className="unit-note">position total</span></div><div className="greek-list">{([['Δ','Delta',analytics.positionGreeks.delta,'Spot direction'],['Γ','Gamma',analytics.positionGreeks.gamma,'Delta curvature'],['Θ','Theta',analytics.positionGreeks.theta,'Annual time decay'],['ν','Vega',analytics.positionGreeks.vega / 100,'Per 1 vol point'],['ρ','Rho',analytics.positionGreeks.rho / 100,'Per 1 rate point']] as const).map(([symbol,label,value,note]) => <div className="greek-row" key={label}><span className="greek-symbol">{symbol}</span><div><strong>{label}</strong><small>{note}</small></div><b className={value >= 0 ? "positive" : "negative"}>{value >= 0 ? "+" : ""}{number(value, 3)}</b></div>)}</div><div className="hedge-callout"><span>Δ</span><div><small>Delta-neutral stock trade</small><strong>{analytics.positionGreeks.delta > 0 ? "Sell" : "Buy"} {number(Math.abs(analytics.positionGreeks.delta), 0)} shares</strong></div></div></article>
        </div>
        <article className="card scenario-card"><div className="card-title"><div><p className="kicker">FULL REVALUATION</p><h3>Scenario analysis</h3></div><span className="unit-note">P&amp;L includes position size</span></div><div className="table-wrap"><table><thead><tr><th>Scenario</th><th>Spot</th><th>Volatility</th><th>Days</th><th>Option value</th><th>Position P&amp;L</th></tr></thead><tbody>{analytics.scenarios.map((s) => <tr key={s.name}><td>{s.name}</td><td>{money(s.spot)}</td><td>{number(s.vol * 100, 1)}%</td><td>{Math.round(s.time * 365)}</td><td>{money(s.value)}</td><td className={s.pnl > 0 ? "positive" : s.pnl < 0 ? "negative" : ""}>{s.pnl > 0 ? "+" : ""}{money(s.pnl)}</td></tr>)}</tbody></table></div></article>
        <article className="card chain-card">
          <div className="card-title"><div><p className="kicker">MARKET VIEW</p><h3>Options chain</h3></div><span className="unit-note">model-generated educational quotes · {chainRows.length} contracts</span></div>
          <div className="chain-toolbar">
            <label><span>Expiration</span><select value={chainExpiration} onChange={(event) => setChainExpiration(Number(event.target.value))}>{[30, 60, 90, 180, 365].map((days) => <option key={days} value={days}>{days} days</option>)}</select></label>
            <label><span>Option type</span><select value={chainType} onChange={(event) => setChainType(event.target.value as "all" | OptionType)}><option value="all">Calls &amp; puts</option><option value="call">Calls only</option><option value="put">Puts only</option></select></label>
            <label><span>Moneyness</span><select value={chainMoneynessFilter} onChange={(event) => setChainMoneynessFilter(event.target.value as "ALL" | "ITM" | "ATM" | "OTM")}><option value="ALL">All</option><option value="ITM">In the money</option><option value="ATM">At the money</option><option value="OTM">Out of the money</option></select></label>
            <div className="chain-legend"><span className="itm-dot" />ITM <span className="atm-dot" />ATM <span className="otm-dot" />OTM</div>
          </div>
          <div className="table-wrap chain-table-wrap">
            <table className="chain-table">
              <thead><tr><th>Type</th><th>Strike</th><th>Money</th><th>Bid</th><th>Ask</th><th>Mid</th><th>Last</th><th>Spread</th><th>Volume</th><th>Open int.</th><th>IV</th><th>Delta</th><th>Gamma</th><th>Theta/day</th><th>Vega/pt</th><th>Rho/pt</th><th>Intrinsic</th><th>Extrinsic</th><th>Theo.</th><th>Market − model</th></tr></thead>
              <tbody>{chainRows.map((row) => <tr key={`${row.days}-${row.strike}-${row.type}`} className={row.moneyness.toLowerCase()}><td><span className={`contract-badge ${row.type}`}>{row.type.toUpperCase()}</span></td><td className="strike-cell">{money(row.strike)}</td><td><span className={`money-badge ${row.moneyness.toLowerCase()}`}>{row.moneyness}</span></td><td>{money(row.bid)}</td><td>{money(row.ask)}</td><td>{money(row.mid)}</td><td>{money(row.last)}</td><td>{money(row.spread)}</td><td>{number(row.volume, 0)}</td><td>{number(row.openInterest, 0)}</td><td>{number(row.iv * 100, 2)}%</td><td className={row.delta >= 0 ? "positive" : "negative"}>{row.delta >= 0 ? "+" : ""}{number(row.delta, 3)}</td><td>{number(row.gamma, 4)}</td><td className={row.theta >= 0 ? "positive" : "negative"}>{number(row.theta, 4)}</td><td>{number(row.vega, 4)}</td><td className={row.rho >= 0 ? "positive" : "negative"}>{number(row.rho, 4)}</td><td>{money(row.intrinsic)}</td><td>{money(row.extrinsic)}</td><td>{money(row.theoretical)}</td><td className={row.edge >= 0 ? "positive" : "negative"}>{row.edge >= 0 ? "+" : ""}{money(row.edge)}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="chain-disclaimer">Bid, ask, last, volume, and open interest are reproducible simulated fields—not live exchange data. Greeks and theoretical values update from the current market inputs.</p>
        </article>
        <div className="advanced-grid">
          <article className="card mc-card"><div className="card-title"><div><p className="kicker">ANALYTICAL VS SIMULATION</p><h3>Monte Carlo validation</h3></div><span className={`validation-badge ${analytics.bs >= analytics.mc.low && analytics.bs <= analytics.mc.high ? "pass" : "review"}`}>{analytics.bs >= analytics.mc.low && analytics.bs <= analytics.mc.high ? "BSM inside 95% CI" : "Review sampling error"}</span></div><div className="validation-metrics"><div><span>MC estimate</span><strong>{money(analytics.mc.price)}</strong></div><div><span>95% confidence interval</span><strong>{money(analytics.mc.low)} — {money(analytics.mc.high)}</strong></div><div><span>Absolute error</span><strong>{money(Math.abs(analytics.mc.price - analytics.bs))}</strong></div><div><span>Standard error</span><strong>{money(analytics.mc.se)}</strong></div></div><div className="mc-convergence">{analytics.mcConvergence.map((row) => <div key={row.simulations}><span>{number(row.simulations, 0)}</span><div><i style={{ width: `${Math.max(3, 100 - Math.min(Math.abs(row.price - analytics.bs) * 200, 96))}%` }} /></div><strong>{money(row.price)}</strong><small>{money(Math.abs(row.price - analytics.bs))} error</small></div>)}</div></article>
          <article className="card surface-card"><div className="card-title"><div><p className="kicker">VOLATILITY</p><h3>Illustrative volatility surface</h3></div><span className="unit-note">skew + smile + term structure</span></div><div className="surface-wrap"><div className="surface-grid"><span className="surface-corner">T \ K</span>{analytics.surfaceStrikes.map((strike) => <span className="surface-axis" key={strike}>{number(strike, 0)}</span>)}{analytics.surface.flatMap((row) => [<span className="surface-axis" key={`m-${row.maturity}`}>{row.maturity}y</span>, ...row.values.map((vol, index) => <span className="surface-cell" style={{ background: `rgba(31,111,104,${Math.min(.2 + vol * 2.1, .9)})`, color: vol > .29 ? "white" : "#142521" }} key={`${row.maturity}-${index}`}>{number(vol * 100, 1)}%</span>)])}</div></div><p className="surface-note">The Python surface module also inverts full market quote grids into implied volatility by strike and maturity.</p></article>
        </div>
        <article className="card hedge-card"><div className="card-title"><div><p className="kicker">TRADING SIMULATION</p><h3>Dynamic delta hedging &amp; hedging P&amp;L</h3></div><span className="unit-note">short one contract · {analytics.hedge.paths} paths · {analytics.hedge.steps} rebalances</span></div><div className="hedge-layout"><div className="hedge-metrics"><div><span>Mean hedge P&amp;L</span><strong className={analytics.hedge.mean >= 0 ? "positive" : "negative"}>{money(analytics.hedge.mean)}</strong></div><div><span>P&amp;L volatility</span><strong>{money(analytics.hedge.std)}</strong></div><div><span>5th percentile</span><strong className="negative">{money(analytics.hedge.p05)}</strong></div><div><span>Median</span><strong>{money(analytics.hedge.median)}</strong></div><div><span>95th percentile</span><strong className="positive">{money(analytics.hedge.p95)}</strong></div><div><span>Implied / realized vol</span><strong>{number(inputs.volatilityPct, 1)}% / {number(inputs.realizedVolPct, 1)}%</strong></div></div><div className="histogram" aria-label="Hedging profit and loss distribution">{analytics.hedge.histogram.map((bin, index) => { const maxCount = Math.max(...analytics.hedge.histogram.map((item) => item.count)); return <div key={index} title={`${money(bin.start)}: ${bin.count} paths`}><i style={{ height: `${Math.max(3, bin.count / maxCount * 100)}%` }} className={bin.start >= 0 ? "gain" : "loss"} /></div>; })}<span className="zero-label">Hedging P&amp;L distribution</span></div></div><div className="hedge-explainer">The simulator sells the option, buys model delta, accrues financing and dividends, pays trading costs, rebalances through time, and settles intrinsic value at expiry.</div></article>
        <div className="bottom-grid">
          <article className="card convergence-card"><div className="card-title"><div><p className="kicker">MODEL CHECK</p><h3>Binomial convergence</h3></div><span className="unit-note">vs {money(analytics.bs)} BSM</span></div><div className="convergence-list">{analytics.convergence.map((row) => { const error = row.value - analytics.bs; return <div key={row.steps}><span>{row.steps} steps</span><div className="bar-track"><i style={{ width: `${Math.max(4, 100 - Math.min(Math.abs(error) * 180, 94))}%` }} /></div><strong>{money(row.value)}</strong><small className={error >= 0 ? "positive" : "negative"}>{error >= 0 ? "+" : ""}{number(error, 4)}</small></div>; })}</div></article>
          <article className="card notes-card"><p className="kicker">INTERPRETATION NOTES</p><h3>What the models assume</h3><ul><li>Black–Scholes and terminal Monte Carlo value European exercise.</li><li>The CRR tree handles early exercise for American contracts.</li><li>The displayed surface is illustrative; market calibration needs option quotes.</li><li>Hedge P&amp;L includes discrete rebalancing and configurable trading costs.</li></ul><div className="parity"><span>Put–call parity carry</span><strong>{money(inputs.spot * Math.exp(-inputs.dividendPct / 100 * inputs.maturity) - inputs.strike * Math.exp(-inputs.ratePct / 100 * inputs.maturity))}</strong></div></article>
        </div>
      </div>
    </section>
    <footer><span>DerivativeLab · Educational analytics engine</span><span>Black–Scholes · CRR · Monte Carlo · Volatility Surface · Dynamic Hedging</span></footer>
  </main>;
}
