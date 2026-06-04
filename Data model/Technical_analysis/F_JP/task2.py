import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

df = pd.read_csv('Nat_Gas.csv')
df.columns = ['Date', 'Price']
df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y')   # FIX: explicit format
prices = df['Price'].values
dates  = df['Date'].values

start_date = date(2020, 10, 31)
end_date   = date(2024, 9,  30)

# Build monthly end-of-month day offsets
months = []
year   = start_date.year
month  = start_date.month + 1
while True:
    current = date(year, month, 1) + timedelta(days=-1)
    months.append(current)
    if current.month == end_date.month and current.year == end_date.year:
        break
    else:
        month = ((month + 1) % 12) or 12
        if month == 1:
            year += 1

days_from_start = [(d - start_date).days for d in months]

def simple_regression(x, y):
    xbar      = np.mean(x)
    ybar      = np.mean(y)
    slope     = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    intercept = ybar - slope * xbar
    return slope, intercept

def bilinear_regression(y, x1, x2):
    slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
    slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
    return slope1, slope2

time             = np.array(days_from_start)
slope, intercept = simple_regression(time, prices)

sin_prices = prices - (time * slope + intercept)
sin_time   = np.sin(time * 2 * np.pi / 365)
cos_time   = np.cos(time * 2 * np.pi / 365)

slope1, slope2 = bilinear_regression(sin_prices, sin_time, cos_time)
amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
shift     = np.arctan2(slope2, slope1)

def interpolate(query_date):

    query_date = pd.Timestamp(query_date)
    days       = (query_date - pd.Timestamp(start_date)).days
    if days in days_from_start:
        return prices[days_from_start.index(days)]
    return (amplitude * np.sin(days * 2 * np.pi / 365 + shift)
            + days * slope + intercept)


def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_storage_volume,
    storage_cost_per_month,
    injection_withdrawal_cost_per_mmbtu,
    transport_cost_per_trade=0.0
):

    injection_dates  = [pd.Timestamp(d) for d in injection_dates]
    withdrawal_dates = [pd.Timestamp(d) for d in withdrawal_dates]

    total_injected  = injection_rate  * len(injection_dates)
    total_withdrawn = withdrawal_rate * len(withdrawal_dates)

    if total_injected > max_storage_volume:
        raise ValueError(
            f"Total injection volume ({total_injected:,.0f} MMBtu) exceeds "
            f"max storage capacity ({max_storage_volume:,.0f} MMBtu)."
        )
    if total_withdrawn > total_injected:
        raise ValueError(
            f"Total withdrawal ({total_withdrawn:,.0f} MMBtu) exceeds "
            f"total injection ({total_injected:,.0f} MMBtu)."
        )

    first_injection    = min(injection_dates)
    last_withdrawal    = max(withdrawal_dates)
    storage_months     = (
        (last_withdrawal.year  - first_injection.year) * 12
        + last_withdrawal.month - first_injection.month
    )
    total_storage_cost = storage_cost_per_month * storage_months

    breakdown      = []
    total_buy_cost = 0.0

    for d in injection_dates:
        price      = interpolate(d)
        buy_cost   = price * injection_rate
        inj_cost   = injection_withdrawal_cost_per_mmbtu * injection_rate
        trans_cost = transport_cost_per_trade
        total_buy_cost += buy_cost
        breakdown.append({
            'date'         : d.strftime('%Y-%m-%d'),
            'action'       : 'INJECT',
            'volume'       : injection_rate,
            'price'        : round(price, 4),
            'trade_value'  : round(-buy_cost, 2),
            'inj_with_cost': round(-inj_cost, 2),
            'transport'    : round(-trans_cost, 2),
            'net_cashflow' : round(-(buy_cost + inj_cost + trans_cost), 2)
        })

    total_sell_revenue = 0.0

    for d in withdrawal_dates:
        price        = interpolate(d)
        sell_revenue = price * withdrawal_rate
        with_cost    = injection_withdrawal_cost_per_mmbtu * withdrawal_rate
        trans_cost   = transport_cost_per_trade
        total_sell_revenue += sell_revenue
        breakdown.append({
            'date'         : d.strftime('%Y-%m-%d'),
            'action'       : 'WITHDRAW',
            'volume'       : withdrawal_rate,
            'price'        : round(price, 4),
            'trade_value'  : round(sell_revenue, 2),
            'inj_with_cost': round(-with_cost, 2),
            'transport'    : round(-trans_cost, 2),
            'net_cashflow' : round(sell_revenue - with_cost - trans_cost, 2)
        })

    total_inj_with_cost  = injection_withdrawal_cost_per_mmbtu * (total_injected + total_withdrawn)
    total_transport_cost = transport_cost_per_trade * (len(injection_dates) + len(withdrawal_dates))

    contract_value = (
        total_sell_revenue
        - total_buy_cost
        - total_storage_cost
        - total_inj_with_cost
        - total_transport_cost
    )

    breakdown.sort(key=lambda x: x['date'])

    return {
        'contract_value'  : round(contract_value,        2),
        'sell_revenue'    : round(total_sell_revenue,    2),
        'buy_cost'        : round(total_buy_cost,        2),
        'storage_cost'    : round(total_storage_cost,    2),
        'inj_with_cost'   : round(total_inj_with_cost,   2),
        'transport_cost'  : round(total_transport_cost,  2),
        'volume_injected' : total_injected,
        'volume_withdrawn': total_withdrawn,
        'breakdown'       : breakdown
    }


def print_result(label, result):
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  Sell revenue       :  ${result['sell_revenue']:>13,.2f}")
    print(f"  Buy cost           :  ${result['buy_cost']:>13,.2f}")
    print(f"  Storage cost       :  ${result['storage_cost']:>13,.2f}")
    print(f"  Inj/With cost      :  ${result['inj_with_cost']:>13,.2f}")
    print(f"  Transport cost     :  ${result['transport_cost']:>13,.2f}")
    print(f"  {'─'*42}")
    print(f"  CONTRACT VALUE     :  ${result['contract_value']:>13,.2f}")
    print(f"\n  Cash flow breakdown:")
    for row in result['breakdown']:
        print(f"    {row['date']}  {row['action']:8s}  "
              f"{row['volume']:>10,.0f} MMBtu @ ${row['price']:.4f}"
              f"  →  net ${row['net_cashflow']:>13,.2f}")


r1 = price_storage_contract(
    injection_dates                     = ['2024-06-30'],
    withdrawal_dates                    = ['2024-10-31'],
    injection_rate                      = 1_000_000,
    withdrawal_rate                     = 1_000_000,
    max_storage_volume                  = 2_000_000,
    storage_cost_per_month              = 100_000,
    injection_withdrawal_cost_per_mmbtu = 0.01,
    transport_cost_per_trade            = 50_000
)
print_result("Test 1 — Single injection (summer) / single withdrawal (autumn)", r1)

r2 = price_storage_contract(
    injection_dates                     = ['2024-04-30', '2024-05-31', '2024-06-30'],
    withdrawal_dates                    = ['2024-11-30', '2024-12-31', '2025-01-31'],
    injection_rate                      = 500_000,
    withdrawal_rate                     = 500_000,
    max_storage_volume                  = 2_000_000,
    storage_cost_per_month              = 100_000,
    injection_withdrawal_cost_per_mmbtu = 0.01,
    transport_cost_per_trade            = 50_000
)
print_result("Test 2 — Three injections (spring) / three withdrawals (winter)", r2)

r3 = price_storage_contract(
    injection_dates                     = ['2025-03-31', '2025-04-30'],
    withdrawal_dates                    = ['2025-11-30', '2025-12-31'],
    injection_rate                      = 750_000,
    withdrawal_rate                     = 750_000,
    max_storage_volume                  = 2_000_000,
    storage_cost_per_month              = 80_000,
    injection_withdrawal_cost_per_mmbtu = 0.005,
    transport_cost_per_trade            = 0
)
print_result("Test 3 — Future extrapolated dates, no transport cost", r3)

continuous_dates = pd.date_range(
    start=pd.Timestamp(start_date),
    end=pd.Timestamp('2025-09-30'),
    freq='D'
)
smoothed_prices = [interpolate(d) for d in continuous_dates]

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(continuous_dates, smoothed_prices, label='Model (historical + forecast)', color='steelblue')
ax.plot(dates, prices, 'o', ms=4, color='navy', label='Observed monthly prices')
ax.axvline(pd.Timestamp(end_date), color='grey', linestyle=':', linewidth=1, label='Forecast start')

ax.axvline(pd.Timestamp('2024-06-30'), color='green', linestyle='--', alpha=0.7, label='Test 1 Buy')
ax.axvline(pd.Timestamp('2024-10-31'), color='red',   linestyle='--', alpha=0.7, label='Test 1 Sell')

ax.set_xlabel('Date')
ax.set_ylabel('Price ($/MMBtu)')
ax.set_title('Natural Gas Price Model with Sample Contract Trade Points')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()  
