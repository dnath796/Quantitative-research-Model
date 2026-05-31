import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Nat_Gas.csv')
df.columns = ['Date', 'Price']
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

prices = df['Price'].values
dates  = df['Date'].values

def get_price(input_date):
   
    input_date = pd.Timestamp(input_date)
    known_days = [d.toordinal() for d in df['Date']]
    query_day  = input_date.toordinal()
    price      = np.interp(query_day, known_days, prices)
    return round(float(price), 2)


def price_contract(
    injection_dates,            # list of dates when we BUY and store gas
    withdrawal_dates,           # list of dates when we WITHDRAW and sell gas
    injection_rate,             # volume bought on each injection date (MMBtu)
    withdrawal_rate,            # volume sold on each withdrawal date (MMBtu)
    max_storage_volume,         # maximum the facility can hold at once (MMBtu)
    storage_cost_per_month,     # monthly rent for the storage facility ($)
    injection_withdrawal_cost,  # cost per MMBtu to inject or withdraw ($)
    transport_cost_per_trade=0  # fixed cost per injection or withdrawal event ($)
):

    injection_dates  = [pd.Timestamp(d) for d in injection_dates]
    withdrawal_dates = [pd.Timestamp(d) for d in withdrawal_dates]

    # --- Volume checks ---
    total_bought = injection_rate  * len(injection_dates)
    total_sold   = withdrawal_rate * len(withdrawal_dates)

    if total_bought > max_storage_volume:
        raise ValueError(f"Total injection ({total_bought:,} MMBtu) exceeds "
                         f"storage capacity ({max_storage_volume:,} MMBtu).")
    if total_sold > total_bought:
        raise ValueError(f"Cannot sell ({total_sold:,} MMBtu) more than "
                         f"was bought ({total_bought:,} MMBtu).")

    # --- Buy cost: price on each injection date x volume ---
    buy_cost = sum(get_price(d) * injection_rate for d in injection_dates)

    # --- Sell revenue: price on each withdrawal date x volume ---
    sell_revenue = sum(get_price(d) * withdrawal_rate for d in withdrawal_dates)

    # --- Storage cost: monthly rent x months between first buy and last sell ---
    first_buy    = min(injection_dates)
    last_sell    = max(withdrawal_dates)
    num_months   = ((last_sell.year  - first_buy.year) * 12
                  +  last_sell.month - first_buy.month)
    storage_cost = storage_cost_per_month * num_months

    # --- Injection/withdrawal cost: per MMBtu moved in AND out ---
    inj_with_cost = injection_withdrawal_cost * (total_bought + total_sold)

    # --- Transport cost: fixed fee per trade event ---
    num_trades     = len(injection_dates) + len(withdrawal_dates)
    transport_cost = transport_cost_per_trade * num_trades

    # --- Final value ---
    contract_value = (sell_revenue
                      - buy_cost
                      - storage_cost
                      - inj_with_cost
                      - transport_cost)

    return round(contract_value, 2)

print("=" * 50)
print("TEST: Single buy in summer, single sell in winter")
print("=" * 50)
val1 = price_contract(
    injection_dates           = ['2024-06-30'],
    withdrawal_dates          = ['2024-10-31'],
    injection_rate            = 1_000_000,
    withdrawal_rate           = 1_000_000,
    max_storage_volume        = 2_000_000,
    storage_cost_per_month    = 100_000,
    injection_withdrawal_cost = 0.01,
    transport_cost_per_trade  = 50_000
)
print(f"Contract Value: ${val1:,.2f}\n")


print("=" * 50)
print("TEST: Multiple buys in spring, multiple sells in winter")
print("=" * 50)
val2 = price_contract(
    injection_dates           = ['2024-04-30', '2024-05-31', '2024-06-30'],
    withdrawal_dates          = ['2024-11-30', '2024-12-31', '2025-01-31'],
    injection_rate            = 500_000,
    withdrawal_rate           = 500_000,
    max_storage_volume        = 2_000_000,
    storage_cost_per_month    = 100_000,
    injection_withdrawal_cost = 0.01,
    transport_cost_per_trade  = 50_000
)
print(f"Contract Value: ${val2:,.2f}\n")


print("=" * 50)
print("TEST: Future dates, no transport cost")
print("=" * 50)
val3 = price_contract(
    injection_dates           = ['2025-03-31', '2025-04-30'],
    withdrawal_dates          = ['2025-11-30', '2025-12-31'],
    injection_rate            = 750_000,
    withdrawal_rate           = 750_000,
    max_storage_volume        = 2_000_000,
    storage_cost_per_month    = 80_000,
    injection_withdrawal_cost = 0.005,
    transport_cost_per_trade  = 0
)
print(f"Contract Value: ${val3:,.2f}\n")


print("=" * 50)
print("TEST: No costs at all (pure buy/sell spread)")
print("=" * 50)
val4 = price_contract(
    injection_dates           = ['2024-06-30'],
    withdrawal_dates          = ['2025-01-31'],
    injection_rate            = 1_000_000,
    withdrawal_rate           = 1_000_000,
    max_storage_volume        = 2_000_000,
    storage_cost_per_month    = 0,
    injection_withdrawal_cost = 0,
    transport_cost_per_trade  = 0
)
print(f"Contract Value: ${val4:,.2f}\n")

future_dates  = pd.date_range(start='2024-10-31', periods=12, freq='MS')
future_prices = [get_price(d) for d in future_dates]

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Price'], marker='o', label='Historical Prices', color='steelblue')
plt.plot(future_dates, future_prices, marker='x', linestyle='--', label='Forecast', color='orange')
plt.axvline(pd.Timestamp('2024-09-30'), color='grey', linestyle=':', label='Forecast start')


plt.axvline(pd.Timestamp('2024-06-30'), color='green', linestyle='--', alpha=0.6, label='Buy (Test 1)')
plt.axvline(pd.Timestamp('2024-10-31'), color='red',   linestyle='--', alpha=0.6, label='Sell (Test 1)')

plt.xlabel('Date')
plt.ylabel('Price ($/MMBtu)')
plt.title('Natural Gas Prices and Sample Contract Trade Points')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

