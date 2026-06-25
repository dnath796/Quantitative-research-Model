import numpy as np 
from scipy.stats import norm 

def calculate_d1(S, K, T, r, sigma):
    numerator = np.log(S / K) + (r + 0.5 * sigma ** 2) * T
    denominator = sigma * np.sqrt(T)
    return numerator / denominator

def calculate_d2(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)
    
    return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

def put_price(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)
    
    return (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def call_delta(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    return norm.cdf(d1)

def put_delta(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def call_theta(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)
    
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    
    return term1 - term2

def call_rho(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)
    
    return K * T * np.exp(-r * T) * norm.cdf(d2)


if __name__ == "__main__":
    price = call_price(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2
    )

    delta = call_delta(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2
    )

    print(f"Call Price: {price:.2f}")
    print(f"Call Delta: {delta:.2f}")