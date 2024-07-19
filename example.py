"""
Simple example of running the model with default configuration values
"""

from abba.model import BankSim


if __name__ == "__main__":

    sim = BankSim()
    sim.run_model()

    print(sim.get_data_frame())
