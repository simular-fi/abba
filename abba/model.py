import dataclasses

import mesa
import numpy as np
import networkx as nx
from tqdm import tqdm

# Agents
from abba.agent import Saver, Bank, Loan

# On init
from abba.logic.f1_init_market import initialize_deposit_base, initialize_loan_book

# On steps
from abba.logic.f2_eval_solvency import main_evaluate_solvency
from abba.logic.f3_second_round_effect import main_second_round_effects
from abba.logic.f4_optimize_risk_weight import main_risk_weight_optimization
from abba.logic.f5_pay_dividends import main_pay_dividends
from abba.logic.f6_expand_loan_book import main_reset_insolvent_loans
from abba.logic.f6_expand_loan_book import main_build_loan_book_locally
from abba.logic.f6_expand_loan_book import main_build_loan_book_globally
from abba.logic.f7_eval_liquidity import main_evaluate_liquidity


@dataclasses.dataclass
class ModelConfig:
    """
    Base model configuration values with default values.
    We use this vs. a Dict for readability and type checking.
    """

    num_steps: int = 10  # num of steps in the model
    num_savers: int = 10_000  # num of savers
    num_loans: int = 20_000  # num of loans
    num_banks: int = 10  # num of banks
    car: float = 0.08  # capital adequacy ratio requirement
    initial_equity: int = 100  # initial bank equity
    risk_free_rate: float = 0.01  # risk free rate
    min_reserves_ratio: float = 0.03  # minimum reserve ratio

    # 0 = bank liquidates loans at face value
    # 1 = fire sale of assets
    bankrupt_liquidation: int = 1


def get_sum_totalassets(model):
    """
    Model reporter
    """
    return sum([x.total_assets for x in model.schedule.agents if isinstance(x, Bank)])


class BankSim(mesa.Model):
    """
    Core model
    """

    # TODO: Using the ModelConfig doesn't work with JupyterViz
    def __init__(
        self,
        num_steps: int = 10,
        num_savers: int = 10_000,
        num_loans: int = 20_000,
        num_banks: int = 10,
        car: float = 0.04,
        initial_equity: int = 100,
        risk_free_rate: float = 0.01,
        min_reserves_ratio: float = 0.03,
        bankrupt_liquidation: int = 1,
    ):
        super().__init__()

        # Model control parameters
        self.num_steps = num_steps
        self.initial_saver = num_savers
        self.initial_loan = num_loans
        self.initial_bank = num_banks

        # Interest rates
        self.rfree = risk_free_rate
        self.libor_rate = risk_free_rate
        self.reserve_rates = risk_free_rate / 2.0

        # Regulatory requirements
        self.car = car
        self.min_reserves_ratio = min_reserves_ratio

        # Bank specific variables
        self.initial_equity = initial_equity
        self.bankrupt_liquidation = bankrupt_liquidation

        # Model setup
        self.G = nx.empty_graph(self.initial_bank)
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)

        # This is only created here to make it easier to get
        # all the column names for the table datacollector below
        b = Bank(
            {
                "unique_id": 0,
                "model": self,
                "equity": 100,
                "rfree": self.rfree,
                "car": self.car,
                "buffer_reserves_ratio": 1.5,
            }
        )
        # Data frames
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={"TotalAssets": get_sum_totalassets},
            tables={"Banks": list(b.into_table_row(0).keys())},
        )

        # Setup agents
        for i in range(self.initial_bank):
            bank = Bank(
                {
                    "unique_id": self.next_id(),
                    "model": self,
                    "equity": 100,
                    "rfree": self.rfree,
                    "car": self.car,
                    "buffer_reserves_ratio": 1.5,
                }
            )
            self.grid.place_agent(bank, i)
            self.schedule.add(bank)

        for i in range(self.initial_saver):
            saver = Saver(
                {
                    "unique_id": self.next_id(),
                    "model": self,
                    "balance": 1,
                    "owns_account": False,
                    "saver_solvent": True,
                    "saver_exit": False,
                    "withdraw_upperbound": 0.2,
                    "exitprob_upperbound": 0.06,
                }
            )
            self.grid.place_agent(saver, np.random.choice(list(self.G.nodes)))
            self.schedule.add(saver)

        for i in range(self.initial_loan):
            loan = Loan(
                {
                    "unique_id": self.next_id(),
                    "model": self,
                    "rfree": self.rfree,
                    "amount": 1,
                    "loan_solvent": True,
                    "loan_approved": False,
                    "loan_dumped": False,
                    "loan_liquidated": False,
                    "pdf_upper": 0.1,
                    "rcvry_rate": 0.4,
                    "firesale_upper": 0.1,
                }
            )
            bank_id = np.random.choice(list(self.G.nodes))
            loan.bank_id = bank_id
            self.grid.place_agent(loan, bank_id)  # Evenly distributed
            self.schedule.add(loan)

        # Initialize model
        initialize_deposit_base(self.schedule)
        initialize_loan_book(self.schedule, self.car, self.min_reserves_ratio)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        # if self.schedule.steps == self.num_steps:
        #    self.running = False

        # evaluate solvency of banks after loans experience default
        main_evaluate_solvency(
            self.schedule, self.reserve_rates, self.bankrupt_liquidation, self.car
        )

        # evaluate second round effects owing to cross_bank linkages
        # only interbank loans to cover shortages in reserves requirements are included
        main_second_round_effects(
            self.schedule, self.bankrupt_liquidation, self.car, self.G
        )

        # Undercapitalized banks undertake risk_weight optimization
        main_risk_weight_optimization(self.schedule, self.car)

        # banks that are well capitalized pay dividends
        main_pay_dividends(self.schedule, self.car, self.min_reserves_ratio)

        # Reset insolvent loans, i.e. rebirth lending opportunity
        main_reset_insolvent_loans(self.schedule)

        # Build up loan book with loans available in bank neighborhood
        main_build_loan_book_locally(self.schedule, self.min_reserves_ratio, self.car)

        # Build up loan book with loans available in other neighborhoods
        main_build_loan_book_globally(self.schedule, self.car, self.min_reserves_ratio)

        # main_raise_deposits_build_loan_book
        # Evaluate liquidity needs related to reserves requirements
        main_evaluate_liquidity(
            self.schedule, self.car, self.min_reserves_ratio, self.bankrupt_liquidation
        )

        self.datacollector.collect(self)
        for b in self.schedule.agents:
            if isinstance(b, Bank):
                # Collect all the bank particulars...
                self.datacollector.add_table_row(
                    "Banks", b.into_table_row(self.schedule.steps)
                )

        self.schedule.step()

    def get_data_frame(self):
        """
        Return the dataframe from the model reporter
        """
        return self.datacollector.get_model_vars_dataframe()

    def get_bank_data_frame(self):
        """
        Return the dataframe from the Bank table
        """
        return self.datacollector.get_table_dataframe("Banks")

    def run_model(self):
        """
        Run the model.
        Use this from scripts or notebooks
        """
        for _ in tqdm(range(self.num_steps)):
            self.step()
