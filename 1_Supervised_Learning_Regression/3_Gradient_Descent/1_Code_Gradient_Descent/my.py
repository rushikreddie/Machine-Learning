
import pandas as pd
import numpy as np

def gradient_descent(x, y,learning_rate=0.01,epochs=3000):
    m,b=0.0,0.0
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)

    for epoch in range(epochs):
        y_pred = m * x_scaled + b
        error = y_scaled - y_pred
        cost = np.mean(error**2)

        dm=-2 * np.mean(error * x_scaled)
        db=-2 * np.mean(error)

        b -= db*learning_rate
        m -= dm*learning_rate

        print(f" m = {m}, b = {b} cost={cost} epoch={epoch}")
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}, b = {b}, m = {m}")

    # Scale back the coefficients to original scale
    b_original = b * (y_max - y_min) + y_min - m * (y_max - y_min) * x_min / (x_max - x_min)
    m_original = m * (y_max - y_min) / (x_max - x_min)

    return b_original, m_original
if '__main__' == __name__:
    df = pd.read_csv("home_prices.csv")

    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()

    b, m = gradient_descent(x, y)

    print(f"Final Results: m={m}, b={b}")


























