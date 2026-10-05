# LEARN — Option Pricing from First Principles

This document teaches the theory behind everything the `dpe` engine
implements: Black-Scholes-Merton and Garman-Kohlhagen (with full
derivations), the Greeks including vanna and volga, implied volatility,
binomial trees, and Monte Carlo with variance reduction. Notation matches
`API_SPEC.md` throughout: spot `s`, strike `k`, time to expiry `t` in years,
volatility `sigma` as a decimal, rates `r` (domestic) and `q` (dividend
yield / foreign rate), continuously compounded. All worked numbers are
golden values from `data/golden/golden.json` — you can reproduce every one
of them with the library in any of the four languages.

---

## 1. The problem, and the one idea that solves it

An option is a right, not an obligation: a European call pays
$\max(S_T - K, 0)$ at expiry $T$, a put pays $\max(K - S_T, 0)$. The naive
approach — discount the *expected* payoff under your real-world forecast of
$S_T$ — is wrong, and understanding *why* it is wrong is the single most
important idea in derivatives pricing.

The reason: the option's payoff can be **replicated** by trading the
underlying and cash. If two portfolios have identical payoffs in every
state of the world, they must have the same price today, or you could sell
the expensive one, buy the cheap one, and lock in riskless profit. The
option's price is therefore the **cost of the replicating strategy** — and
that cost turns out not to depend on anyone's forecast of where the stock
is going. The expected return of the stock, the quantity everyone argues
about, cancels out of the price. What remains is the volatility.

Everything else in this document is machinery for computing that
replication cost: in closed form (Black-Scholes-Merton), on a lattice
(binomial trees), or by simulation (Monte Carlo).

---

## 2. The model: geometric Brownian motion

Black-Scholes-Merton (BSM) assumes the underlying follows geometric
Brownian motion (GBM) with constant coefficients under the real-world
measure $\mathbb{P}$:

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
$$

with a continuous dividend yield $q$ (the holder of the stock receives
$q S_t\,dt$ in dividends per unit time), a constant risk-free rate $r$, no
transaction costs, no market impact, continuous trading, and unlimited
borrowing/shorting at $r$. By Itô's lemma the log-spot is arithmetic
Brownian motion, so the solution is

$$
S_T = S_0 \exp\!\Big[\big(\mu - \tfrac{1}{2}\sigma^2\big)T + \sigma W_T\Big],
$$

i.e. $\ln S_T$ is normal — $S_T$ is lognormal. The $-\tfrac{1}{2}\sigma^2$
is the Itô correction: the mean of a lognormal exceeds the exponential of
the mean of its log, and this term is exactly what makes
$\mathbb{E}[S_T] = S_0 e^{\mu T}$ come out right.

Where these assumptions break in real life:

* **Constant volatility** is the big one. Market option prices imply a
  volatility that varies with strike and maturity (the *smile*/*skew*,
  §5.4), which is the market telling you returns are not lognormal: fat
  tails, and (for equities) crash-heavier left tails.
* **Continuous frictionless hedging** — real hedging is discrete and costs
  money; the replication is approximate and the residual risk is real
  (gamma losses between rebalances).
* **Constant rates** are a fine approximation for short-dated equity
  options, less so for multi-year FX or rates products.
* **No jumps** — earnings gaps, devaluations and de-pegs are jumps; no
  continuous hedge can replicate through a jump.
* **Continuous dividend yield** is a modelling convenience. Index options
  are well described by a yield; single stocks pay discrete cash
  dividends, which matter for American calls in particular.

The model survives despite all this because it is used as a *quoting and
hedging framework*, not as a literal description: traders quote implied
vols, hedge BSM deltas, and manage the model's errors with the higher-order
Greeks (§6).

---

## 3. Deriving Black-Scholes-Merton

### 3.1 Route one: replication and the BSM PDE

Let $V(S, t^\ast)$ be the option value as a function of spot and calendar
time $t^\ast$. Form a portfolio: long the option, short $\Delta$ units of
stock. By Itô's lemma,

$$
dV = \Big( V_{t^\ast} + \mu S V_S + \tfrac{1}{2}\sigma^2 S^2 V_{SS} \Big) dt
     + \sigma S V_S \, dW .
$$

The portfolio $\Pi = V - \Delta S$ has dynamics (remembering the short
stock position also *pays out* the dividend yield, costing $\Delta q S\,dt$):

$$
d\Pi = dV - \Delta\, dS - \Delta q S\, dt .
$$

Choose $\Delta = V_S$. The $dW$ terms cancel **exactly** — the portfolio is
instantaneously riskless. A riskless portfolio must earn the riskless rate
($d\Pi = r \Pi\, dt$), otherwise arbitrage. Substituting and cancelling the
$\mu$ terms (this is where the real-world drift disappears) gives the
**Black-Scholes-Merton PDE**:

$$
V_{t^\ast} + \tfrac{1}{2}\sigma^2 S^2 V_{SS} + (r - q) S V_S - r V = 0 ,
$$

solved backwards from the terminal condition
$V(S, T) = \max(\pm(S - K), 0)$. The drift $\mu$ appears nowhere: two
traders who disagree violently about the stock's prospects must still agree
on the option price, because both can hedge it.

### 3.2 Route two: risk-neutral pricing

The modern restatement: absence of arbitrage is (essentially) equivalent to
the existence of a measure $\mathbb{Q}$ under which discounted
*self-financing* wealth is a martingale. Under $\mathbb{Q}$ (via Girsanov's
theorem, which shifts the drift of $W$ but leaves $\sigma$ untouched) the
stock's drift becomes $r - q$:

$$
dS_t = (r - q) S_t\, dt + \sigma S_t\, dW^{\mathbb{Q}}_t ,
\qquad
V_0 = e^{-rT}\, \mathbb{E}^{\mathbb{Q}}\big[\,\text{payoff}(S_T)\,\big].
$$

The two routes are the same fact seen from two sides: the PDE is the
Feynman-Kac representation of the expectation. Route two is what the Monte
Carlo engine implements literally: simulate under $\mathbb{Q}$, average the
payoff, discount.

### 3.3 Evaluating the expectation → the formula

Under $\mathbb{Q}$, $\ln S_T \sim \mathcal{N}\big(\ln s + (r - q -
\tfrac{1}{2}\sigma^2)t,\ \sigma^2 t\big)$. Computing
$e^{-rt}\mathbb{E}^{\mathbb{Q}}[(S_T - k)^+]$ splits into two lognormal
partial expectations, and each evaluates to a normal CDF term:

$$
\boxed{\;
\text{call} = s\,e^{-qt} N(d_1) - k\,e^{-rt} N(d_2), \qquad
\text{put} = k\,e^{-rt} N(-d_2) - s\,e^{-qt} N(-d_1)
\;}
$$

$$
d_1 = \frac{\ln(s/k) + (r - q + \tfrac{1}{2}\sigma^2)\,t}{\sigma\sqrt{t}},
\qquad
d_2 = d_1 - \sigma\sqrt{t}.
$$

How to read it: $N(d_2)$ is the risk-neutral probability the call finishes
in the money, so $k e^{-rt} N(d_2)$ is the discounted expected strike
payment. $s e^{-qt} N(d_1)$ is the discounted expected value of the stock
*received given exercise* — $d_1 > d_2$ because, conditional on ending in
the money, $S_T$ is on average higher than unconditionally. Equivalently,
$e^{-qt}N(d_1)$ is the replicating stock holding — the delta.

**Put-call parity** follows model-free from
$(S_T - K)^+ - (K - S_T)^+ = S_T - K$:

$$
C - P = s\,e^{-qt} - k\,e^{-rt}.
$$

Any implementation must satisfy it to machine precision; the test suites
check it over a grid of $(s, k, t)$.

### 3.4 Worked example (golden case `bs_call_atm`)

$s = 100$, $k = 100$, $t = 1$, $\sigma = 0.20$, $r = 0.05$, $q = 0$:

$$
d_1 = \frac{0 + (0.05 + 0.02)\cdot 1}{0.20} = 0.35, \qquad
d_2 = 0.35 - 0.20 = 0.15
$$

$N(d_1) = 0.636831$, $N(d_2) = 0.559618$:

$$
C = 100 \times 0.636831 - 100\, e^{-0.05} \times 0.559618
  = 63.6831 - 53.2325 = \mathbf{10.45058357}.
$$

Parity gives the put without re-deriving anything:
$P = C - (s - k e^{-rt}) = 10.45058 - 4.87706 = \mathbf{5.57352602}$
(golden case `bs_put_atm`).

### 3.5 Garman-Kohlhagen: FX is BSM with `q = rf`

An FX rate $S$ quoted **domestic-per-foreign** (e.g. USD per EUR ≈ 1.10 for
EURUSD) is the price of an asset — one unit of foreign currency — that pays
a continuous yield: parked in a foreign money-market account it grows at
the foreign rate $r_f$. So the foreign rate plays *exactly* the role of a
dividend yield, and the FX option formula (Garman & Kohlhagen, 1983) is BSM
with $r = r_d$, $q = r_f$:

$$
C = s\,e^{-r_f t} N(d_1) - k\,e^{-r_d t} N(d_2),
\qquad
d_1 = \frac{\ln(s/k) + (r_d - r_f + \tfrac{1}{2}\sigma^2) t}{\sigma\sqrt{t}}.
$$

The price is in **domestic currency per unit of foreign notional**. The
forward comes from covered interest parity,
$F = s\,e^{(r_d - r_f)t}$ — in the engine, `fx_forward`.

Worked example (golden case `gk_call_eurusd`): EURUSD call, $s = 1.10$,
$k = 1.12$, $t = 0.5$, $\sigma = 0.10$, $r_d = 3\%$, $r_f = 2\%$. Then
$d_1 = -0.148754$, $d_2 = -0.219465$ and the price is
$\mathbf{0.02430351}$ USD per EUR of notional — 243 USD pips. On
€1,000,000 notional the premium is \$24,303.51. Negative rates need no
special treatment: `gk_put_negative_rate` prices with $r_d = -0.5\%$
(JPY/CHF-style) and only the discount factors change.

### 3.6 Degenerate limits — part of the contract

The formulas above assume $t > 0$, $\sigma > 0$, $k > 0$; the limits are
handled exactly, not by plugging tiny epsilons in:

* $t = 0$: intrinsic value $\max(\pm(s - k), 0)$, no discounting.
* $\sigma = 0$, $t > 0$: the terminal spot is the deterministic forward
  $F = s e^{(r-q)t}$, so the value is the **discounted forward intrinsic**
  $e^{-rt}\max(\pm(F - k), 0) = \max(\pm(s e^{-qt} - k e^{-rt}), 0)$.
* $k = 0$: a zero-strike call is a forward claim on the asset, worth
  $s e^{-qt}$; a zero-strike put is worthless.

Getting these wrong is the classic source of NaNs in production risk
systems ($\ln(s/k)$ and division by $\sigma\sqrt{t}$ both blow up).

---

## 4. The Greeks

The Greeks are the partial derivatives of $V$ — the sensitivities a hedger
actually trades. Units in this engine (fixed across all four languages):
theta per **year** of calendar time; vega, vanna, volga per **unit** of vol
(per 1.00 = 100 vol points); rho per unit of rate.

$$
\begin{aligned}
\Delta_{\text{call}} &= e^{-qt} N(d_1)
&\qquad \Delta_{\text{put}} &= e^{-qt}\,(N(d_1) - 1)\\[2pt]
\Gamma &= \frac{e^{-qt}\,\varphi(d_1)}{s\,\sigma\sqrt{t}}
&\qquad \text{vega} &= s\,e^{-qt}\,\varphi(d_1)\sqrt{t}\\[2pt]
\Theta_{\text{call}} &= -\frac{s e^{-qt}\varphi(d_1)\sigma}{2\sqrt{t}}
 - r k e^{-rt}N(d_2) + q s e^{-qt}N(d_1)
&\qquad \rho_{\text{call}} &= k\,t\,e^{-rt}N(d_2)
\end{aligned}
$$

with $\varphi$ the standard normal pdf; gamma and vega are identical for
calls and puts (parity: $C - P$ is linear in $s$, so all second derivatives
and all vol derivatives agree).

Intuition, one by one:

* **Delta** — the replicating stock holding, and (approximately, via
  $N(d_2)$) the moneyness probability. ATM ≈ 0.5–0.6, deep ITM → $e^{-qt}$,
  deep OTM → 0.
* **Gamma** — how fast delta moves; the cost of being wrong between hedge
  rebalances. Peaks near ATM and explodes as $t \to 0$ for ATM options
  (see the weekly put in the demo: $\Gamma = 0.095$ vs 0.019 for the 1y
  ATM). Long options are long gamma: rebalancing systematically buys low
  and sells high, financed by theta.
* **Theta** — the rent paid for gamma. The first term
  $-s e^{-qt}\varphi(d_1)\sigma / (2\sqrt{t})$ *is* the gamma rent: note
  $\tfrac{1}{2}\sigma^2 s^2 \Gamma$ equals it exactly. The BSM PDE
  rearranged says $\Theta + \tfrac{1}{2}\sigma^2 s^2 \Gamma + (r-q) s
  \Delta = rV$: theta and gamma are two sides of the same coin.
* **Vega** — exposure to the (implied) vol level. Grows like $\sqrt{t}$
  for ATM options, so long-dated options are the vol instruments (LEAPS
  vega 65 vs weekly 4.6 in the demo).
* **Rho** — rate exposure; largest for long-dated ITM options. For FX,
  `rho` here is with respect to the **domestic** rate $r_d$ only.

### 4.1 Vanna and volga — second-order vol risk

$$
\text{vanna} = \frac{\partial^2 V}{\partial s\,\partial\sigma}
             = -e^{-qt}\varphi(d_1)\frac{d_2}{\sigma},
\qquad
\text{volga} = \frac{\partial^2 V}{\partial \sigma^2}
             = \text{vega}\cdot\frac{d_1 d_2}{\sigma}.
$$

* **Vanna** is how delta responds to vol (equivalently, how vega responds
  to spot). It changes sign at $d_2 = 0$ (strike at the forward's
  "median"): for an OTM call, more vol pushes delta up; for a sufficiently
  ITM one, more vol pushes delta back toward $e^{-qt}/2$. Vanna is the
  Greek of the **skew**: a book hedged in delta and vega but long vanna
  gains when spot and vol move together.
* **Volga** ("vol gamma", vomma) is the convexity of price in vol —
  positive when $d_1 d_2 > 0$ (away from the money on either side), near
  zero ATM. Volga is the Greek of the **smile**: OTM strangles have volga,
  ATM straddles have almost none, which is precisely why the market charges
  extra for wings (§5.4). FX desks quote risk reversals (vanna) and
  butterflies (volga) as the smile's coordinates; the *vanna-volga* method
  prices exotics by charging for exactly these two exposures.

Worked numbers (golden `bs_greeks_call_atm`, same inputs as §3.4):
$\Delta = 0.63683065$, $\Gamma = 0.01876202$, vega $= 37.52403469$
($= 100 \cdot \varphi(0.35) \cdot 1$; per vol *point* divide by 100 →
0.375 per 1%), $\Theta = -6.41402755$ per year (−1.76 cents/day),
$\rho = 53.23248155$, vanna $= -0.28143026$
($= -\varphi(0.35)\cdot 0.15/0.20$), volga $= 9.85005911$
($= 37.524 \times 0.35 \times 0.15 / 0.20$ — small: ATM options are nearly
linear in vol, which is also why vega hedging ATM is stable).

In the degenerate region ($t=0$, $\sigma=0$ or $k=0$) all the
curvature/vol Greeks are exactly zero and delta/theta/rho collapse to the
derivatives of the limiting price — spelled out in `API_SPEC.md` §3.2 so
that all four languages agree bit-for-bit at the boundaries.

---

## 5. Implied volatility

### 5.1 What it means

The BSM price is **strictly increasing in $\sigma$** (vega > 0 for
$t, \sigma, k > 0$), so on the arbitrage-admissible price interval the map
$\sigma \mapsto \text{price}$ is invertible: every admissible market price
corresponds to exactly one **implied volatility**. Implied vol is the
market's price *re-expressed in vol units* — a normalisation that strips
out moneyness, maturity and discounting so that different options become
comparable. It is a *forward-looking, risk-neutral* quantity: it embeds
both the market's variance forecast and the risk premium for bearing
volatility, which is why implied typically sits above subsequently realised
vol (the variance risk premium), and why comparing it with
`historical_vol` output is a trade idea, not an identity.

Vol is also the market's **quoting convention**: an FX desk quotes "9.85/10.05
vol" rather than a premium — see the FX recipes in `COOKBOOK.md`.

### 5.2 No-arbitrage bounds first

Model-free bounds for a call:
$\max(s e^{-qt} - k e^{-rt}, 0) \le C \le s e^{-qt}$ (mirrored for puts
with upper bound $k e^{-rt}$). A price at or outside these bounds has **no
implied vol** — the correct behaviour is a clear error naming the violated
bound, not a root-finder grinding to 0 or 2000%. `implied_vol` checks the
bounds before any iteration (and the demo shows the rejection message for a
below-intrinsic price).

### 5.3 The solver: safeguarded Newton in a bracket

Newton with the analytic vega converges quadratically near the root — but
raw Newton is dangerous exactly where users need implied vol most: deep
ITM/OTM, where vega is tiny and one Newton step can catapult $\sigma$
to $10^{6}$ or negative. The engine's contract (identical in all four
languages):

1. maintain a **bracket** $[lo, hi]$ that provably contains the root
   (monotonicity makes this trivial: model price too low ⇒ root above);
   the initial `hi` doubles from 1.0 until the model price exceeds the
   target, clipped to and tested at the 2000% cap (a price the model
   cannot reach even at the cap is an error);
2. take the Newton step **only if it lands strictly inside the bracket**
   and vega is not degenerate; otherwise **bisect**;
3. stop when $|\text{model} - \text{price}| < \texttt{tol}$ (default
   $10^{-10}$) or the bracket is narrower than $10^{-12}$ in vol;
4. if neither happens within `max_iter` steps, **raise** — the residual and
   bracket go in the message, and no number is returned. `tol` must be
   positive and `max_iter` at least 1; the engine validates both.

Bisection guarantees convergence; Newton, when usable, supplies the speed;
rule 4 guarantees that a returned vol always satisfies a stopping rule (an
earlier revision fell off the end of the loop and returned whatever
$\sigma$ it was holding — with `max_iter = 1` that was 0.198 for a true
0.200, silently). Golden cases `iv_roundtrip_*` verify
$\sigma \to \text{price} \to \sigma$ round-trips to $10^{-6}$, including
OTM cases where the first iterations are bisections. Note the tolerance is
*absolute in price*: for a sub-pip FX wing $10^{-10}$ is only $10^{-6}$
relative, and for an index quoted in the thousands the price cannot be
matched to $10^{-10}$ at all — there the bracket-width rule terminates the
solve with an essentially exact vol.

### 5.4 The smile — what inverting BSM reveals

Invert market prices across strikes at one expiry and you do not get a
constant: equity indices show a downward **skew** (low strikes richer —
crash protection), FX shows a **smile** (both wings rich). This is not a
numerical artifact; it is the market pricing non-lognormal tails. Each
$(k, t)$ gets its own $\sigma_{\text{imp}}(k, t)$, the *implied vol
surface*, and BSM survives as the coordinate system in which the surface is
expressed. FX surfaces are quoted at deltas (25Δ risk reversal / butterfly)
rather than strikes — which is why the delta-convention helpers in §8
matter for anything touching an FX surface.

---

## 6. Binomial trees

### 6.1 One period: replication made discrete

Let the spot move to $su$ or $sd$ over one step of length $\Delta t$, and
let the option pay $V_u$, $V_d$. Solve for a portfolio of $\Delta$ stock
plus cash $B$ matching both outcomes:

$$
\Delta = \frac{V_u - V_d}{s(u - d)}, \qquad
V = e^{-r\Delta t}\big[p V_u + (1 - p) V_d\big], \qquad
p = \frac{e^{(r-q)\Delta t} - d}{u - d}.
$$

The value is the discounted expectation under the **risk-neutral
probability** $p$ — not any real-world probability. This is §3's whole
story in two lines of algebra, which is why the tree is the best teaching
model in finance.

For $p$ to be a probability we need $d < e^{(r-q)\Delta t} < u$; if the
drift outruns the vol spacing at a coarse step size the lattice is invalid,
and the engine raises an invalid-input error rather than silently clamping.

### 6.2 CRR and JR parameterisations

Both discretise GBM with $\Delta t = t/n$ and match its log-moments to
$O(\Delta t)$; they differ in where they put the drift.

* **Cox-Ross-Rubinstein (CRR)**: $u = e^{\sigma\sqrt{\Delta t}}$,
  $d = 1/u$, $p$ as above. Since $ud = 1$ the lattice **recombines around
  the spot** — the middle node at every even level is exactly $s$, which is
  what makes tree Greeks natural (§6.5).
* **Jarrow-Rudd (JR)**: with $\nu = (r - q - \tfrac{1}{2}\sigma^2)\Delta t$,
  $u = e^{\nu + \sigma\sqrt{\Delta t}}$, $d = e^{\nu - \sigma\sqrt{\Delta t}}$,
  $p = \tfrac{1}{2}$. The drift lives in the node placement and both
  branches are equally likely; the lattice drifts with the forward.

Sanity check with $n = 3$ on the §3.4 call: $\Delta t = 1/3$,
$u = 1.12240$, $d = 0.89095$, $p = 0.54378$; rolling back gives 11.044 —
crude at 3 steps, but already the right shape, and by $n = 2000$ CRR gives
10.44958378 (golden `crr_euro_call_2000`), within $10^{-3}$ of the BS
10.45058.

### 6.3 Convergence — and the odd/even oscillation

For European payoffs both trees converge to BSM at rate $O(1/n)$, but not
monotonically: the error **oscillates** between adjacent step counts as the
strike's position shifts relative to the terminal node grid. The demo
table shows CRR errors $-3.99\times10^{-2}$ at $n=50$ shrinking to
$-1.00\times10^{-3}$ at $n=2000$ — roughly $\propto 1/n$.

**Richardson averaging** (`richardson=true`) prices at $n$ and $n+1$ steps
and averages. Because the leading error term flips sign between odd and
even $n$, the average cancels it: the demo shows the $n=50$ error dropping
from $4\times10^{-2}$ to $2.7\times10^{-3}$ — one to two extra digits for
~2× the work. Note this two-point averaging targets the *oscillating* term;
it is cheap insurance, not a high-order extrapolation scheme, and it does
not apply cleanly to American options where the early-exercise boundary
moves with $n$.

### 6.4 American exercise

An American option is an **optimal stopping problem**, and backward
induction solves the discrete version exactly:

$$
V(i, j) = \max\Big(\underbrace{e^{-r\Delta t}\big[pV(i{+}1,j{+}1) + (1{-}p)V(i{+}1,j)\big]}_{\text{continuation}},\ \underbrace{\text{intrinsic}(S(i,j))}_{\text{exercise}}\Big)
$$

applied at **every** node. Where intrinsic wins, the holder exercises; the
locus of such nodes approximates the early-exercise boundary.

When is early exercise optimal? For a **put**, killing the position
harvests interest on the strike ($rK$) at the cost of the remaining
optionality — deep ITM with $r > 0$, exercise wins. Hence the premium in
the demo table grows with strike: from 0.16 at $k = 90$ to 2.74 at
$k = 120$ ($s = 100$). For a **call on a non-dividend stock** early
exercise is *never* optimal ($C_{\text{amer}} = C_{\text{euro}}$: selling
always beats exercising, since the European call is worth more than
intrinsic when $q=0$, $r\ge0$); with dividends ($q > 0$) it can be, to
capture the yield. Golden anchors: `crr_amer_put_500` = 6.08881011 vs the
European 5.57352602, and `crr_amer_put_div_500` shows the dividend pushing
the American put premium around. Tests assert American ≥ European
everywhere and premium monotone in $k$.

### 6.5 Greeks from the tree

Re-pricing with bumped inputs costs extra tree builds and adds bump noise;
delta, gamma and theta can instead be read off **one** backward induction
by retaining the first three levels ($V(i,j)$, $S(i,j)$):

$$
\Delta = \frac{V(1,1) - V(1,0)}{S(1,1) - S(1,0)}, \qquad
\Gamma = \frac{\dfrac{V(2,2)-V(2,1)}{S(2,2)-S(2,1)} - \dfrac{V(2,1)-V(2,0)}{S(2,1)-S(2,0)}}{\tfrac{1}{2}\big(S(2,2) - S(2,0)\big)}
$$

For theta, compare the middle level-2 node with the root — two time steps
apart. For CRR, $S(2,1) = s$ exactly and
$\Theta = (V(2,1) - V(0,0))/(2\Delta t)$ is a pure calendar difference. For
JR the lattice drifts ($S(2,1) = s e^{2\nu}$), so the engine subtracts the
delta/gamma contribution of the spot displacement
$\delta s = S(2,1) - s$:

$$
\Theta = \frac{V(2,1) - V(0,0) - \Delta\,\delta s - \tfrac{1}{2}\Gamma\,\delta s^2}{2\Delta t}.
$$

Golden `crr_greeks_call_500` pins delta 0.63676685, gamma 0.01879379,
theta −6.42023444 (stored at full double precision in `golden.json`).
Delta and gamma sit within $10^{-4}$ of the analytic 0.63683065 and
0.01876202; theta is $6\times10^{-3}$ off the analytic −6.41402755 —
an order of magnitude worse, because the calendar difference is a
one-sided $O(\Delta t)$ estimate over two steps of width $1/500$. The
same construction works unchanged for American options where no analytic
Greeks exist. The smallest legal lattice is `steps = 2`, where level 2 *is*
the terminal level; the engine stores it explicitly so the formulas above
still apply (a case the contract allows and every port tests).

---

## 7. Monte Carlo

### 7.1 Exact GBM stepping

GBM has a closed-form transition, so the simulation uses the **exact**
step

$$
S_{t+h} = S_t \exp\!\Big[(r - q - \tfrac{1}{2}\sigma^2)h + \sigma\sqrt{h}\,Z\Big],
\qquad Z \sim \mathcal{N}(0,1),
$$

which is exact **in distribution at the grid dates** — no Euler bias, any
step size. A European vanilla therefore needs exactly one step to $t$.
What remains biased is anything that looks *between* grid dates:
discretely-monitored barriers and lookbacks (§7.5).

### 7.2 Standard errors — the honest part of MC

The estimator is a sample mean of i.i.d. discounted payoffs; by CLT,

$$
\widehat{V} \pm z_{0.975}\,\frac{\widehat{\text{sd}}}{\sqrt{n}},
\qquad z_{0.975} = 1.959963984540054 .
$$

Every MC function returns an `MCResult` with the point estimate, the
standard error and this 95% CI. Two non-negotiable rules the engine bakes
in:

* the error decays like $O(1/\sqrt{n})$ — one more digit costs 100× the
  paths, which is *why* variance reduction exists;
* with antithetic variates the raw paths are **not independent**: the
  i.i.d. unit is the **pair average** $(f(Z_i) + f(-Z_i))/2$, and the SE is
  computed over the $n/2$ pair averages. Computing it over raw paths
  understates the error when pairs are positively correlated and is a real
  and common bug — the cross-language statistics contract (`API_SPEC.md`
  §5) exists to rule it out.

### 7.3 Antithetic variates

Pair each draw $Z$ with $-Z$. For a payoff monotone in $Z$ (vanillas,
Asians in each fixing), $f(Z)$ and $f(-Z)$ are negatively correlated, so
the pair average has variance
$\tfrac{1}{2}\mathrm{Var}(f)(1 + \rho)$ with $\rho < 0$ — same cost,
lower variance, and odd-moment sampling error cancels exactly. Demo, 100k
paths: SE 0.0468 plain → 0.0331 antithetic.

### 7.4 Control variates

If a variable $X$ correlated with the payoff $Y$ has **known mean**
$\mathbb{E}[X]$, estimate instead

$$
Y - \beta\,(X - \mathbb{E}[X]), \qquad
\beta^\ast = \frac{\mathrm{Cov}(Y, X)}{\mathrm{Var}(X)}
\;\Rightarrow\;
\mathrm{Var} \to \mathrm{Var}(Y)\,(1 - \rho_{XY}^2).
$$

$\beta$ is estimated in-sample (an $O(1/n)$ bias, negligible at these path
counts; $\beta = 0$ if the control is degenerate). Controls used:

* **European vanilla**: the discounted terminal spot $e^{-rt}S_T$, a
  martingale with known mean $s e^{-qt}$ (the $k=0$ BS value — "BS analytic
  as control"). Demo (100k paths): SE 0.0468 plain → 0.0331 antithetic →
  0.0088 antithetic + control, i.e. variance reductions of
  $(0.0331/0.0088)^2 \approx 14\times$ vs antithetic and
  $(0.0468/0.0088)^2 \approx 28\times$ vs plain at the same path count.
* **Arithmetic Asian**: the **geometric-average** Asian payoff on the same
  paths. The geometric mean of lognormals is lognormal, so the discrete
  geometric Asian has a closed form (Black's formula on
  $\mathbb{E}[\ln G] = \ln s + (r - q - \tfrac{\sigma^2}{2})\,t\frac{n+1}{2n}$,
  $\mathrm{Var}[\ln G] = \sigma^2 t \frac{(n+1)(2n+1)}{6n^2}$; golden
  `geo_asian_call_analytic` = 5.94020022 at 12 fixings). Arithmetic and
  geometric averages correlate at $\rho > 0.99$, so the control removes
  ~99.8% of the variance: the demo's arithmetic Asian SE is 0.0008 at 100k
  paths — a plain estimator would need roughly $400\times$ the paths for
  the same accuracy.

### 7.5 Discretisation bias: barriers and lookbacks

The up-and-out barrier is monitored on the grid only; a continuous barrier
can be breached *between* dates, which discrete monitoring never sees. So
discrete monitoring **over-prices** an up-and-out relative to the
continuous contract with $O(1/\sqrt{n_{\text{steps}}})$ bias; the lookback
is the mirror image (grid extrema understate continuous extrema →
under-pricing). The engine deliberately prices the **discrete** contract
and applies no correction; if a continuous contract is intended, the
Broadie-Glasserman-Kou barrier shift $B \to B\,e^{0.5826\,\sigma\sqrt{h}}$
removes the leading bias term. Know which contract you are pricing — the
term sheet says.

### 7.6 Monte Carlo Greeks

Naive finite differences with *independent* runs at $s \pm h$ have variance
$O(1/h^2)$ — useless. Two proper methods, both implemented:

* **Pathwise derivative**: differentiate the payoff along the path. For
  GBM, $\partial S_T/\partial S_0 = S_T/S_0$, so for a call
  $\Delta = e^{-rt}\,\mathbb{E}[\mathbf{1}\{S_T > k\}\, S_T/s]$. Unbiased
  and low variance — valid because the vanilla payoff is Lipschitz
  (interchange of derivative and expectation is legal). It **fails for
  discontinuous payoffs** (digitals, barriers at the knockout), where the
  likelihood-ratio method is the standard alternative.
* **Finite differences with common random numbers (CRN)**: re-price at
  $s(1 \pm h)$ reusing the **same** normal draws and difference *per path*.
  The two prices' sampling errors cancel path by path, leaving the variance
  bounded as $h \to 0$ (for Lipschitz payoffs). CRN generalises to any
  payoff and any Greek at the cost of the usual $O(h^2)$ FD bias.

Demo, ATM call: analytic 0.6368, pathwise 0.6379 (SE 0.0006), CRN-FD
0.6378 (SE 0.0006) — both within two standard errors of truth.

---

## 8. Market conventions: equity vs FX

| | Equity | FX |
|---|---|---|
| Underlying | stock/index price | rate, **domestic per foreign** (EURUSD = USD per EUR) |
| Carry input | dividend yield $q$ | foreign rate $r_f$ (as $q$), domestic $r_d$ (as $r$) |
| Premium | domestic cash | domestic ccy per unit foreign notional (pips) — or % foreign |
| Quotes | price or implied vol per strike | **vol by delta**: ATM, 25Δ/10Δ risk reversal & butterfly |
| Surface axis | strike / moneyness | delta |
| Typical smile | downward skew | symmetric-ish smile, skewed by rate/crash asymmetry |

**Delta conventions.** "25-delta" only means something once you fix *which*
delta. This engine implements the two premium-excluded ("pips")
conventions:

* **spot delta** $\;e^{-r_f t} N(d_1)$ — hedge in spot; what `gk_greeks`
  reports;
* **forward delta** $\;N(d_1)$ — hedge in forwards; standard for longer
  maturities.

They differ by a pure discounting factor (one forward on one unit of
foreign currency has spot sensitivity $e^{-r_f t}$), so the conversion is
payoff- and strike-independent and sign-preserving:

$$
\Delta_{\text{fwd}} = e^{+r_f t}\,\Delta_{\text{spot}}, \qquad
\Delta_{\text{spot}} = e^{-r_f t}\,\Delta_{\text{fwd}} .
$$

Golden `gk_call_eurusd`: spot delta 0.43648706, forward delta 0.44087382
($= N(d_1)$ exactly). Mixing conventions when interpolating a
delta-quoted surface shifts every strike you back out — a silent,
expensive bug. *Premium-included* deltas (premium paid in foreign
currency, standard for e.g. USD-quoted EM pairs) are a further wrinkle,
out of scope here.

Other conventions worth knowing: `t` is a year fraction (ACT/365-style)
and the engine never touches calendars — day-count is the caller's job;
theta is per year (divide by 365 for per-day); vega per unit of vol
(divide by 100 for per-vol-point).

---

## 9. Common pitfalls and numerical issues

1. **Tail cancellation in $N(x)$.** $0.5(1 + \mathrm{erf}(x/\sqrt2))$
   loses all relative precision for $x \ll 0$ (catastrophic cancellation);
   the engine mandates $N(x) = \tfrac{1}{2}\mathrm{erfc}(-x/\sqrt2)$, and
   golden case `bs_call_deep_otm` enforces it at tolerance $10^{-10}$.
2. **Degenerate limits by epsilon.** Plugging $t = 10^{-300}$ instead of
   handling $t = 0$ yields NaN via $0 \times \infty$. Handle the limits
   exactly (§3.6); tests pin them.
3. **Newton without a bracket** diverges on deep ITM/OTM implied vols
   (tiny vega ⇒ huge steps). Always safeguard with a maintained bracket
   (§5.3), check no-arb bounds *before* iterating, and make running out of
   iterations an *error*, not a return value — a 2-vol-point miss that
   raises nothing poisons a whole surface.
4. **Tree probability outside (0,1)** when $(r - q)\Delta t$ outruns
   $\sigma\sqrt{\Delta t}$ (coarse steps, high drift, low vol). Error out;
   never clamp silently. The same goes for **lattice overflow**: $s u^n$
   with $\sigma\sqrt{t n} \gtrsim 700$ is `inf` in double precision (and
   `inf · 0` in $s u^j d^{n-j}$ is NaN); the engine rejects such lattices
   up front.
4b. **Seeds and integer types.** A seed that means different things in
   different ports (silently wrapped to $2^{64}-1$ in one, rejected by the
   RNG library in another) breaks reconciliation; the contract pins
   $[0, 2^{63}-1]$ everywhere. In Python, `isinstance(x, int)` is `False`
   for `numpy.int64`, so batch code driven from a DataFrame must be
   accepted via `numbers.Integral` — the engine does this.
5. **SE over raw antithetic paths** understates the error (§7.2). The
   i.i.d. unit is the pair average.
6. **Comparing MC across languages by seed.** RNG streams differ by
   library; the contract fixes *statistical* tolerances for MC goldens and
   bit-level determinism only within a language.
7. **Discrete vs continuous monitoring.** A 100-step barrier price is not
   the continuous-barrier price (§7.5). Know the contract; correct or
   refine deliberately.
8. **Theta sign and scale confusion.** This engine: calendar-time
   derivative, per year, usually negative for long options. Desk systems
   often report per-day theta — a factor-365 mismatch looks like a bug but
   is a convention.
9. **Vega scaling.** Per unit vol here; per vol point (÷100) on most
   desks. Cross-language golden values are the arbiter.
10. **Pathwise Greeks on discontinuous payoffs** are silently wrong
    (the derivative misses the point mass at the kink/jump); use
    likelihood-ratio or CRN-FD instead.
11. **Oscillating tree convergence**: judging accuracy from one step count
    is misleading ($n$ vs $n+1$ can straddle the true price); use
    Richardson averaging or look at adjacent counts.
12. **Historical vs implied vol confusion**: `historical_vol` is a
    backward-looking realised estimate (ddof = 1 on log returns,
    annualised by $\sqrt{252}$); implied vol is forward-looking and
    risk-neutral. They should differ; the gap is the variance risk premium.

---

## 10. Interview-style Q&A

**Q1. Why doesn't the stock's expected return appear in the BSM price?**
Because the option is replicated by continuously trading stock and cash;
delta-hedging cancels the $dW$ *and* the $\mu$ terms (§3.1). The price is
the replication cost, identical for a bull and a bear. Formally: under the
pricing measure the drift is forced to $r - q$ regardless of $\mu$.

**Q2. What are $N(d_1)$ and $N(d_2)$?**
$N(d_2)$ = risk-neutral probability of finishing ITM. $e^{-qt}N(d_1)$ =
the delta, and $s e^{-qt} N(d_1)$ = discounted expected stock value
*conditional on exercise* times the exercise probability. $d_1 = d_2 +
\sigma\sqrt t$ because conditioning on ITM tilts the lognormal upward.

**Q3. Your delta and vega are hedged. What kills you?**
Second order: gamma (spot convexity — realised vs implied vol between
rebalances), vanna (spot-vol cross: delta becomes wrong when vol moves)
and volga (vol convexity: vega becomes wrong when vol moves). A book flat
in delta/vega but short wings is short volga and gets hurt in a vol spike;
short vanna hurts when spot and vol move together (equity sell-offs).

**Q4. Why is early exercise never optimal for an American call on a
non-dividend stock, but sometimes optimal for a put?**
Alive, the call is worth more than intrinsic ($C \ge s - k e^{-rt} >
s - k$ for $r>0$): selling beats exercising, so waiting dominates. The put
is the asymmetric case: exercising ITM releases the strike $K$ *now*,
earning interest $rK$, and the deeper ITM the less remaining optionality
you give up — so for $r > 0$ there is a critical spot below which
exercising wins. With dividends the call story changes: exercise just
before a dividend can capture it.

**Q5. A trader quotes an option price below its intrinsic value. What does
your implied-vol solver do, and why?**
It raises an error naming the violated no-arbitrage lower bound before
any iteration (§5.2). No $\sigma \ge 0$ can produce that price, so
root-finding is meaningless — and returning 0 or NaN would poison
downstream risk. (In the market you would question the quote: stale price,
wrong forward, or an arbitrage.)

**Q6. Why does the binomial price oscillate as you add steps, and what do
you do about it?**
The terminal node grid shifts relative to the strike as $n$ changes, so the
lattice's effective strike placement error flips sign between odd and even
$n$. Averaging the $n$ and $n{+}1$ prices (two-point Richardson) cancels
the oscillating leading term — the demo shows ~15× error reduction at
$n = 50$.

**Q7. Your MC price is 10.42 with SE 0.05 against an analytic 10.45. Is
the implementation wrong?**
No — the error is 0.6 standard errors; ~55% of correct runs land further
away. Worry outside 3 SE, and *verify* by increasing paths (error should
shrink like $1/\sqrt n$) or switching seeds. This is exactly why every
estimate carries an SE and the MC golden tolerances are several SE wide.

**Q8. Explain antithetic variates and when they can fail to help.**
Pair $Z$ with $-Z$; for payoffs monotone in $Z$ the pair members are
negatively correlated so the pair-average variance drops below half the
plain variance (§7.3). For non-monotone payoffs (straddle-like: high at
both tails) $f(Z)$ and $f(-Z)$ can be *positively* correlated and
antithetics can be worse than plain sampling. Also remember the SE must be
computed on pair averages.

**Q9. Why use the geometric Asian as a control variate for the arithmetic
Asian rather than the underlying's terminal value?**
Effectiveness is $1 - \rho^2$. The geometric average correlates with the
arithmetic average at $\rho > 0.99$ (same fixings, means differ only by
AM-GM convexity), giving ~99%+ variance reduction, and its discrete-fixing
price is known in closed form because a product of lognormals is lognormal
(§7.4). The terminal spot correlates far less with an average-rate payoff.

**Q10. What is the difference between spot delta and forward delta for an
FX option, and when does it matter?**
Spot delta $e^{-r_f t}N(d_1)$ hedges in spot; forward delta $N(d_1)$
hedges in forwards; they differ by the deterministic factor $e^{r_f t}$
(§8). It matters whenever strikes are backed out of a delta-quoted vol
surface (25Δ RR/BF): using the wrong convention — spot vs forward, and
premium-included vs excluded — silently shifts every strike, most severely
for long maturities and high rate differentials.

---

## 11. Further reading

Full citations (journal, volume, pages, DOI) for every work the code
implements are in `README.md` § References; the list below is the reading
order.

* J. Hull, *Options, Futures, and Other Derivatives* — the standard
  first pass at BSM, trees, Greeks and FX options.
* S. Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*
  — the replication and risk-neutral arguments of §3 made rigorous
  (Girsanov, martingale representation, Feynman-Kac).
* P. Glasserman, *Monte Carlo Methods in Financial Engineering* — the
  canonical reference for §7: variance reduction, pathwise vs
  likelihood-ratio Greeks, discretisation bias.
* J. Cox, S. Ross & M. Rubinstein (1979), "Option Pricing: A Simplified
  Approach", *JFE* — the CRR tree paper.
* R. Jarrow & A. Rudd (1983), *Option Pricing* — the equal-probability
  lattice.
* M. Garman & S. Kohlhagen (1983), "Foreign Currency Option Values",
  *J. Int. Money & Finance* — FX as dividend-yield BSM.
* M. Broadie, P. Glasserman & S. Kou (1997), "A Continuity Correction for
  Discrete Barrier Options", *Mathematical Finance* — the 0.5826 barrier
  shift of §7.5.
* U. Wystup, *FX Options and Structured Products* — FX quoting and delta
  conventions (§8) in full, premium-included deltas, vanna-volga pricing.
* P. Jäckel, "Let's Be Rational" (2015) — how far implied-vol inversion
  can be pushed beyond §5.3's safeguarded Newton.
* E. Haug & N. Taleb, "Option Traders Use (Very) Sophisticated Heuristics,
  Never the Black-Scholes-Merton Formula" — a provocation on what the
  formula is actually *for*; read after §2's assumptions list.

---

## Numerical engineering in the C++ port

The equations in this document are exact mathematical identities, but production
numerical code must also define behavior at floating-point boundaries. The C++
port therefore treats degenerate regions (`t = 0`, `sigma = 0`, `k = 0`)
explicitly instead of forcing them through formulas containing division by
`√t`, `sigma` or `k`. At the deterministic ATM boundary it uses a
scale-aware machine-epsilon tolerance before applying the documented one-sided
delta tie-break. This prevents a value that is mathematically zero from being
misclassified because subtraction leaves a residual on the order of `1e-15`.

The engineering workflow is also part of reproducibility: CMake builds the C++17
library and tests, GoogleTest/CTest exercises unit and golden cases, ASan+UBSan
can run the suite under runtime checks, Google Benchmark provides microbenchmarks,
and profiling-friendly `RelWithDebInfo` builds preserve useful stack traces for
Linux `perf` and macOS Instruments. None of these tools changes the financial
model; they make implementations of the model easier to verify and measure.

See [`cpp/README.md`](cpp/README.md) for commands.
