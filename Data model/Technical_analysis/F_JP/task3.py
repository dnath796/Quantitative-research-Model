import pandas as pd
import numpy as np

df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

features = ['credit_lines_outstanding', 'loan_amt_outstanding',
            'total_debt_outstanding', 'income',
            'years_employed', 'fico_score']

X = df[features].values   # convert to plain numpy array
y = df['default'].values  # 0 or 1


np.random.seed(42)
indices    = np.random.permutation(len(X))
split      = int(0.8 * len(X))
train_idx  = indices[:split]
test_idx   = indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled  = (X_test  - mean) / std


def sigmoid(z):
    """Converts any number into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate=0.1, epochs=1000):
    """
    Trains logistic regression using gradient descent.

    At each step:
      - Make a prediction using current weights
      - Measure how wrong we are (the error)
      - Nudge the weights in the direction that reduces the error
    """
    n_samples, n_features = X.shape

    # Start weights at zero
    weights = np.zeros(n_features)
    bias    = 0.0

    for epoch in range(epochs):
        # Forward pass: predict probabilities
        z          = X @ weights + bias   # linear combination
        y_pred     = sigmoid(z)           # convert to probability

        # Compute gradients (how much each weight contributed to the error)
        error      = y_pred - y
        dw         = (X.T @ error) / n_samples
        db         = error.mean()

        # Update weights
        weights   -= learning_rate * dw
        bias      -= learning_rate * db

    return weights, bias

def predict_proba(X, weights, bias):
    """Returns probability of default for each row."""
    return sigmoid(X @ weights + bias)

# Train the model
weights, bias = train_logistic_regression(X_train_scaled, y_train)


proba      = predict_proba(X_test_scaled, weights, bias)
y_hat      = (proba >= 0.5).astype(int)   # 0.5 threshold
correct    = (y_hat == y_test).sum()
accuracy   = correct / len(y_test)

print(f"Test Accuracy: {accuracy*100:.2f}%")


RECOVERY_RATE = 0.10
LGD           = 1 - RECOVERY_RATE   # 90%

def expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):
    """
    Estimates the expected loss on a loan.

    Expected Loss = PD x LGD x EAD
      PD  = probability of default (our logistic regression)
      LGD = 1 - recovery rate = 0.90
      EAD = loan amount outstanding

    Parameters
    ----------
    credit_lines_outstanding : int   — number of active credit lines
    loan_amt_outstanding     : float — current loan balance ($)
    total_debt_outstanding   : float — total debt across all obligations ($)
    income                   : float — annual income ($)
    years_employed           : int   — years at current employer
    fico_score               : int   — FICO credit score (300-850)

    Returns
    -------
    dict:
        pd  : probability of default (0-1)
        lgd : loss given default (0.90)
        ead : exposure at default ($)
        el  : expected loss ($)
    """
    # Build the feature vector and scale using training mean/std
    borrower        = np.array([credit_lines_outstanding,
                                loan_amt_outstanding,
                                total_debt_outstanding,
                                income,
                                years_employed,
                                fico_score], dtype=float)
    borrower_scaled = (borrower - mean) / std

    # Predict probability of default
    pd_value = float(sigmoid(borrower_scaled @ weights + bias))

    ead = loan_amt_outstanding
    el  = pd_value * LGD * ead

    return {
        'pd'  : round(pd_value, 4),
        'lgd' : LGD,
        'ead' : round(ead, 2),
        'el'  : round(el, 2)
    }


test_cases = [
    ('Low Risk',    dict(credit_lines_outstanding=0, loan_amt_outstanding=5000,
                         total_debt_outstanding=2000,  income=80000,
                         years_employed=5, fico_score=720)),
    ('Medium Risk', dict(credit_lines_outstanding=2, loan_amt_outstanding=8000,
                         total_debt_outstanding=6000,  income=45000,
                         years_employed=3, fico_score=620)),
    ('High Risk',   dict(credit_lines_outstanding=5, loan_amt_outstanding=15000,
                         total_debt_outstanding=20000, income=22000,
                         years_employed=1, fico_score=480)),
]

for label, case in test_cases:
    result = expected_loss(**case)
    print(f"\n{label}")
    print(f"  Probability of Default : {result['pd']*100:.2f}%")
    print(f"  Loan Amount (EAD)      : ${result['ead']:,.2f}")
    print(f"  Loss Given Default     : {result['lgd']*100:.0f}%")
    print(f"  Expected Loss          : ${result['el']:,.2f}")
