import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def generate_lending_data(n=10000, seed=42):
    np.random.seed(seed)

    loan_amnt = np.random.lognormal(mean=9.5, sigma=0.6, size=n).clip(1000, 40000).round(0)
    term = np.random.choice([36, 60], size=n, p=[0.72, 0.28])
    int_rate = np.random.uniform(5.0, 28.0, n).round(2)
    installment = (loan_amnt * (int_rate / 1200) /
                   (1 - (1 + int_rate / 1200) ** -term)).round(2)
    grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
    grade_idx = np.clip(((int_rate - 5) / 3.5).astype(int), 0, 6)
    grade = np.array([grade_map[g] for g in grade_idx])
    emp_length = np.random.choice(range(0, 11), size=n,
                                  p=[0.08, 0.12, 0.10, 0.09, 0.08,
                                     0.09, 0.08, 0.07, 0.07, 0.08, 0.14])
    annual_inc = np.random.lognormal(mean=11.0, sigma=0.6, size=n).clip(10000, 300000).round(0)
    dti = np.random.uniform(0, 40, n).round(2)
    home_ownership = np.random.choice(
        ['RENT', 'OWN', 'MORTGAGE'], size=n, p=[0.40, 0.10, 0.50])
    purpose = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement',
         'major_purchase', 'small_business', 'car', 'medical', 'other'],
        size=n, p=[0.45, 0.15, 0.10, 0.05, 0.05, 0.08, 0.04, 0.08])
    open_acc = np.random.poisson(lam=11, size=n).clip(1, 50)
    revol_bal = np.random.lognormal(mean=9.0, sigma=1.0, size=n).clip(0, 200000).round(0)
    revol_util = np.random.uniform(0, 100, n).round(1)
    total_acc = (open_acc + np.random.poisson(lam=10, size=n)).clip(2, 80)
    delinq_2yrs = np.random.poisson(lam=0.3, size=n).clip(0, 10)
    pub_rec = np.random.poisson(lam=0.1, size=n).clip(0, 5)
    mort_acc = np.random.poisson(lam=1.5, size=n).clip(0, 15)

    logit = (-4.0
             + 0.15 * (int_rate - 12)
             + 0.8 * (dti - 15) / 10
             - 0.6 * np.log(annual_inc / 60000)
             + 0.8 * (home_ownership == 'RENT').astype(float)
             + 0.6 * delinq_2yrs
             + 0.8 * pub_rec
             - 0.2 * emp_length / 5
             + 0.4 * (term == 60).astype(float)
             + 0.6 * (purpose == 'small_business').astype(float)
             + np.random.normal(0, 0.3, n))
    prob = 1 / (1 + np.exp(-logit))
    default = (np.random.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate,
        'installment': installment, 'grade': grade,
        'emp_length': emp_length, 'annual_inc': annual_inc,
        'dti': dti, 'home_ownership': home_ownership, 'purpose': purpose,
        'open_acc': open_acc, 'revol_bal': revol_bal,
        'revol_util': revol_util, 'total_acc': total_acc,
        'delinq_2yrs': delinq_2yrs, 'pub_rec': pub_rec,
        'mort_acc': mort_acc, 'default': default
    })
    return df


def save_data(df=None):
    if df is None:
        df = generate_lending_data()
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, 'lending_club.csv')
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print(f"Default rate: {df['default'].mean():.1%}")
    return path


if __name__ == '__main__':
    save_data()
