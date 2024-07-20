import random
from abba.agent import Bank, Saver, Loan

"""
NOTE: Setup can be slow based on the size of the experiment.
"""


def initialize_deposit_base(schedule):
    """
    Connect the savers (customers) with the bank based on
    grid location created during model.init().  Then, update the banks deposits/reserves
    """
    print("... initialize deposit base ...")

    for bank in [x for x in schedule.agents if isinstance(x, Bank)]:
        savers = [
            x for x in schedule.agents if isinstance(x, Saver) and x.pos == bank.pos
        ]
        for saver in savers:
            saver.bank_id = bank.pos
            saver.owns_account = True
        bank.bank_deposits = sum([x.balance for x in savers])
        bank.bank_reserves = bank.bank_deposits + bank.equity


def initialize_loan_book(schedule, car, min_reserves_ratio):
    """
    Connects banks to loans.  This affects the bank's risk-weighted assets
    This could use some work to determine the best way to do it as it's slow...
    """
    print("... initializing loan book ...")
    print(
        " ** this can take a while if there are a large number of configured loans **"
    )

    for bank in [x for x in schedule.agents if isinstance(x, Bank)]:
        bank.bank_reserves = bank.equity + bank.bank_deposits
        bank.calculate_reserve_ratio()
        bank.max_rwa = bank.equity / (1.1 * car)

        interim_reserves = bank.bank_reserves
        interim_deposits = bank.bank_deposits
        interim_reserves_ratio = bank.reserves_ratio

        rwa = 0
        unit_loan = 0  # cumulative amount in loans

        available_loans = True

        # get all available (unapproved) loans and process
        # this is intensive loop if the number of loans is large
        while (
            available_loans
            and rwa < bank.max_rwa
            and interim_reserves_ratio > bank.buffer_reserves_ratio * min_reserves_ratio
        ):

            pool_loans = [
                x
                for x in schedule.agents
                if isinstance(x, Loan) and bank.pos == x.pos and not x.loan_approved
            ]
            available_loans = len(pool_loans) > 0

            if available_loans:
                # note: loans are not removed from the pool, just filtered (above) based on 'approved'...
                loan = random.choice(pool_loans)
                interim_reserves = interim_reserves - loan.amount
                interim_reserves_ratio = (
                    interim_reserves / interim_deposits if interim_deposits != 0 else 0
                )
                loan.loan_approved = True
                unit_loan = unit_loan + loan.amount
                rwa = rwa + loan.rweight * loan.amount

        bank.bank_loans = unit_loan
        bank.rwassets = rwa
        bank.bank_reserves = bank.bank_deposits + bank.equity - bank.bank_loans
        bank.calculate_reserve_ratio()
        bank.calculate_capital_ratio()
        bank.bank_provisions = sum(
            [
                x.pdef * x.lgdamount
                for x in schedule.agents
                if isinstance(x, Loan)
                and bank.pos == x.pos
                and x.loan_approved
                and x.loan_solvent
            ]
        )
        bank.bank_solvent = True
        bank.calculate_total_assets()
        bank.calculate_leverage_ratio()
