"""
Simple example of running the model with default configuration values
"""

from abba.model import BankSim


if __name__ == "__main__":

    sim = BankSim(num_steps=5)
    sim.run_model()
    bank_data = sim.get_bank_data_frame()
    bank_data.loc[:, "reserve_ratio"] = bank_data[bank_data.deposits > 0].apply(
        lambda x: x["reserves"] / x["deposits"], axis=1
    )

    plotit = (
        bank_data[["step", "reserves", "deposits", "reserve_ratio"]]
        .groupby(["step"])
        .mean()
    )

    print(sim.get_data_frame())
    print(bank_data)
    print(plotit)
