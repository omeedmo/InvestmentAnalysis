"""
One vocabulary, every consumer.

Every surface in this app answers questions about the same handful of ideas —
revenue, debt, net income — but until now each kept its own list of the XBRL
concepts that spell them. The analyze page had `METRIC_TAGS`; the screener had
four named lists plus a scatter of literals passed inline to `_merge_frames`;
the quarterly pass had its own copy of which metrics are balance-sheet
instants; the UI had row keys that matched none of them by construction; and
each company template carried its own concepts plus a `point_in_time` list
that was a third copy of the same flow-or-balance question.
Nothing checked any list against any other, so a concept added to one was
simply absent from the rest until a filer happened to need it, and the gap
surfaced as a blank cell someone noticed months later:

    ProfitLoss            in the screener, not the app  -> VeriSign had no net
                          income for 13 years
    SeniorNotes           in the screener, not the app  -> VeriSign had no debt
                          at all, and no EV
    pretax_income         no tags anywhere in the app, and no UI row either, so
                          the absence was invisible rather than blank

This module is the single place those decisions live. Consumers ask it what
they need and hold no concept literals of their own.


ROLE — why a flat shared list would have been worse than the drift
──────────────────────────────────────────────────────────────────
The consumers do not resolve a candidate list the same way, so "the tags for
net income" is not a well-formed question until you say what the tag is FOR:

    SERIES     the candidate list proper, priority-ordered. `extract_annual_-
               series` keeps the largest absolute value across the whole list;
               `extract_point_in_time_series` and the screener's `_merge_frames`
               take the first that resolves.

    GAP_FILL   applied ONLY where the series list resolved nothing, per period.
               This is the role a flat list cannot express, and getting it
               wrong is not a blank cell but a wrong number: ProfitLoss is the
               consolidated figure and NetIncomeLoss the parent-attributable
               one, so putting ProfitLoss in the series list lets it outrank
               NetIncomeLoss under the largest-absolute-value rule for every
               filer that HAS minority interests — Brandywine's FY2025 net
               income would move from -178,247 to -178,867 thousand. As a
               gap-fill it fixes VeriSign and touches nobody else.
               On a first-hit-wins consumer a gap-fill is just the tail of the
               resolution order, which is why the screener could get away with
               one flat list and the analyze page could not.

    SIGNED     the same measure under the opposite sign convention, so a
               consumer must normalize before using it and the plain gap-fill
               pass must NOT pick it up. Interest expense is the case: filers
               that stop tagging InterestExpense report a combined net-interest
               figure instead, negative-for-expense, against the positive
               convention every consumer downstream assumes.

    PROXY      a DIFFERENT measure standing in for this one. Never applied by
               a generic pass — a consumer has to name it and say why it is
               tolerable there. A weighted-average share count is not a
               period-end count, but it is close enough to size a market cap on
               a screen, and close enough to fill a quarter for a filer (META)
               that tags neither share-count concept at all. It would not be
               close enough for the annual column, which is why nothing applies
               it there.

    COMPONENT  a part of a total, summed only when no total resolved. Realty
               Income prints revolving credit, term loans, mortgages payable
               and notes payable as four lines totalling ~$28.8B and tags no
               consolidated debt at all; a total always wins over the sum, so a
               filer tagging both is not double counted.

SURFACE — where a concept is deliberately not shared
────────────────────────────────────────────────────
`on` records, per concept, which surfaces use it and in what role. A concept
absent from a surface is now a decision written down here rather than an
omission invisible in a second file. The recurring reason is scale: the
analyze page reads one filer and can afford a nine-deep candidate list, while
the screener reads a `frames` slice across ~3,700 filers at once and a
loose tag there mis-prices the whole screen. `capex` is the clearest case —
nine concepts on the analyze side, two on the screen.

KIND — flow or instant
──────────────────────
A property of the concept, not of the surface, though the two extraction
paths in app.py had drifted into disagreeing about `equity_securities`
(see its note below). Consumers read `kind` rather than keeping a set.

TEMPLATES — extending this for one filer
────────────────────────────────────────
A company template may add element names to a metric for its own filer, which
is the mechanism that lets a coverage gap real for one company stay noise for
the next. An element name nothing here has heard of is exactly what belongs in
one. Reaching past a decision made here is not, and `template_conflicts()`
refuses it: a concept declared screen-only, or declared as a component or a
gap-fill rather than a candidate, would undo on one filer the distinction that
keeps a row meaning one thing. What a template no longer decides is `kind` —
a metric it introduces is declared here with `by_template=True`, so the annual
and quarterly columns cannot classify it differently.

Run `python3 tag_audit.py` to check that no consumer has grown a literal of
its own again.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ─── Surfaces ────────────────────────────────────────────────────────────────
ANALYZE = "analyze"   # app.py: the per-filer annual and quarterly passes
SCREEN  = "screen"    # screener.py: SEC `frames` + the companyfacts fallback
BOTH    = (ANALYZE, SCREEN)

# ─── Roles ───────────────────────────────────────────────────────────────────
SERIES    = "series"
GAP_FILL  = "gap_fill"
COMPONENT = "component"
SIGNED    = "signed"      # same measure, opposite sign convention — see below
PROXY     = "proxy"       # a DIFFERENT measure standing in — see below

# ─── Namespaces ──────────────────────────────────────────────────────────────
GAAP = "us-gaap"
IFRS = "ifrs-full"
DEI  = "dei"

# ─── Kinds ───────────────────────────────────────────────────────────────────
FLOW    = "flow"       # duration fact: a period's income / cash movement
INSTANT = "instant"    # point-in-time fact: a balance-sheet position


@dataclass(frozen=True)
class Concept:
    """One XBRL element name, and what each surface does with it."""
    tag: str
    ns: str = GAAP
    on: dict = field(default_factory=lambda: {ANALYZE: SERIES, SCREEN: SERIES})

    def role(self, surface: str):
        return self.on.get(surface)


@dataclass(frozen=True)
class Metric:
    """One idea, spelled by concepts, rendered (or deliberately not) as a row."""
    kind: str
    concepts: tuple
    quarterly: bool = False   # extracted for the quarter columns as well
    row: bool = True          # has a UI row; False must carry a `note` saying why
    by_template: bool = False # concepts come per filer from company_templates
    note: str = ""


# ─── Shorthand used by the vocabulary below ──────────────────────────────────
# A bare string is the common case: a us-gaap concept used as a series entry by
# both surfaces. Everything else says how it differs.

def A(tag, role=SERIES, ns=GAAP):
    """Analyze-only. The screener deliberately does not read this concept."""
    return Concept(tag, ns, {ANALYZE: role})


def S(tag, role=SERIES, ns=GAAP):
    """Screen-only."""
    return Concept(tag, ns, {SCREEN: role})


def G(tag, on=BOTH, ns=GAAP):
    """Gap-fill: used only where the series list resolved nothing."""
    return Concept(tag, ns, {s: GAP_FILL for s in on})


def P(tag, on=(SCREEN,), ns=GAAP):
    """Component of a total: summed only when no total resolved."""
    return Concept(tag, ns, {s: COMPONENT for s in on})


def IFRSC(tag, role=SERIES):
    """An ifrs-full concept. Reaches the screener's companyfacts fallback only —
    the analyze extractors read us-gaap and dei, so 20-F filers never get here."""
    return Concept(tag, IFRS, {SCREEN: role})


def DEIC(tag, on=BOTH, role=SERIES):
    return Concept(tag, DEI, {s: role for s in on})


def M(kind, concepts, **kw):
    return Metric(kind,
                  tuple(c if isinstance(c, Concept) else Concept(c) for c in concepts),
                  **kw)


# ═════════════════════════════════════════════════════════════════════════════
# The vocabulary
# ═════════════════════════════════════════════════════════════════════════════

VOCAB: dict[str, Metric] = {

    # ── Income statement ────────────────────────────────────────────────────
    # The screen reads revenue only to rank filers into the Fortune-500-shaped
    # universe, and takes the largest across the list, so the analyze-only
    # entries are the ones that would move a ranking without measuring anything
    # better — IncludingAssessedTax carries sales taxes the other spellings
    # exclude, and the three narrow captions rarely appear without one of the
    # broad ones alongside.
    "revenue": M(FLOW, [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        A("RevenueFromContractWithCustomerIncludingAssessedTax"),
        "Revenues", "SalesRevenueNet", A("SalesRevenueGoodsNet"),
        A("SalesRevenueServicesNet"), A("NetSales"),
    ], quarterly=True),

    "cost_of_revenue": M(FLOW, [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",  # AMR and similar mining/industrial
    ], quarterly=True, row=False,
        note="Feeds gross profit, which is the row; shown as a margin, not a line."),

    "gross_profit": M(FLOW, ["GrossProfit"], quarterly=True),

    # Split cost-of-revenue components: some filers (e.g. INTU pre-2018) tag
    # CostOfGoodsSold and CostOfServices separately with no consolidated total.
    "cost_of_goods_component": M(FLOW, ["CostOfGoodsSold"], row=False,
        note="Half of a split cost of revenue; only ever read to rebuild the total."),
    "cost_of_services_component": M(FLOW, ["CostOfServices"], row=False,
        note="Half of a split cost of revenue; only ever read to rebuild the total."),

    "rd_expense":  M(FLOW, ["ResearchAndDevelopmentExpense"], quarterly=True),
    "sga_expense": M(FLOW, ["SellingGeneralAndAdministrativeExpense"], quarterly=True),

    # G&A as a separate line (useful for REIT NOI derivation: NOI = EBITDA + G&A)
    "general_admin_expense": M(FLOW, [
        "GeneralAndAdministrativeExpense",
        # Some companies embed G&A in SGA — only use as fallback when G&A is not separately filed
    ], quarterly=True, row=False,
        note="Read for the REIT NOI derivation (NOI = EBITDA + G&A), which is the row."),

    # Selling & marketing expense (filed separately from G&A by some companies, e.g. TMHC post-2021)
    "selling_marketing_expense": M(FLOW, [
        "SellingAndMarketingExpense",
        "SellingExpense",
    ], quarterly=True, row=False,
        note="Operating-expense line read only for the gross-profit add-back fallback."),

    # Operating expense lines used for the gross-profit add-back fallback
    # (GP = OI + S&M + R&D + G&A + amortization + restructuring; e.g. INTU post-2018)
    "amortization_of_intangibles": M(FLOW, ["AmortizationOfIntangibleAssets"],
        quarterly=True, row=False,
        note="Operating-expense line read only for the gross-profit add-back fallback."),
    "restructuring_charges": M(FLOW, ["RestructuringCharges", "RestructuringCosts"],
        quarterly=True, row=False,
        note="Operating-expense line read only for the gross-profit add-back fallback."),

    "operating_income": M(FLOW, [
        "OperatingIncomeLoss",
        # Fallback for companies (e.g. BRK) that don't separately file OperatingIncomeLoss
        # but report pre-tax earnings — closest available proxy for conglomerates/insurers.
        #
        # Analyze-only, and deliberately so. Substituting a pre-tax figure for
        # an operating one is a per-filer judgement you can check by opening the
        # statements; applied across a `frames` slice of ~3,700 filers it would
        # quietly reprice every EV/EBIT on the screen with a different measure.
        A("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
        A("IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
        A("IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"),
        IFRSC("ProfitLossFromOperatingActivities"),
    ], quarterly=True),

    "interest_expense": M(FLOW, [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestExpenseNonoperating",   # MCK FY2025+ and similar
        "InterestAndDebtExpense",
        # Some filers (e.g. CHTR from FY2014) stop tagging a dedicated interest
        # expense and report only a combined "net interest income/expense"
        # figure, signed negative-for-expense against the positive convention
        # every consumer here assumes. SIGNED rather than GAP_FILL because the
        # value has to have abs() applied and only where it is negative — a
        # plain gap-fill would write a credit balance into an expense row.
        Concept("InterestIncomeExpenseNet", GAAP, {ANALYZE: SIGNED}),
        Concept("InterestIncomeExpenseNonoperatingNet", GAAP, {ANALYZE: SIGNED}),
    ], quarterly=True),

    "income_tax": M(FLOW, ["IncomeTaxExpenseBenefit"], quarterly=True),

    "net_income": M(FLOW, [
        "NetIncomeLoss",
        A("NetIncomeLossAvailableToCommonStockholdersBasic"),
        # A filer with no noncontrolling interests has nothing separating
        # consolidated profit from the parent's share, and some drop the parent
        # concept entirely — VeriSign from FY2013. GAP_FILL, not series: under
        # the largest-absolute-value rule a series entry would let the
        # consolidated figure outrank the parent-attributable one for every
        # filer that DOES have minority interests (BDN FY2025: -178,247 ->
        # -178,867 thousand). The screener resolves first-hit-wins, so for it a
        # gap-fill is simply the tail of the list — which is exactly what its
        # own flat ["NetIncomeLoss", "ProfitLoss"] always was.
        G("ProfitLoss"),
        IFRSC("ProfitLoss", GAP_FILL),
    ], quarterly=True),

    # Pretax income had no entry here at all, so it was never read from
    # companyfacts: a bound filer picked it up from the binding overlay and
    # every unbound one showed an empty row. It is NOT operating income —
    # VeriSign's FY2025 operating income is $1,121.0M against pretax income of
    # $1,068.5M, the $52.5M difference being net interest on its senior notes.
    # The check that the right concept is bound: $1,068.5M less the $242.8M tax
    # provision is $825.7M, its net income to the dollar.
    "pretax_income": M(FLOW, [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ], quarterly=True),

    # ── Bank-specific income statement metrics ──────────────────────────────
    "interest_income": M(FLOW, [
        "InterestAndDividendIncomeOperating", "InterestAndFeeIncomeLoansAndLeases",
        "InterestIncomeOperating",
    ], quarterly=True),
    "net_interest_income": M(FLOW, [
        "InterestIncomeExpenseNet", "InterestIncomeExpenseAfterProvisionForLosses",
    ], quarterly=True),
    "noninterest_income":  M(FLOW, ["NoninterestIncome"], quarterly=True),
    # The screen reads this one as a membership test rather than a measurement:
    # a filer that tags non-interest expense at all is a bank.
    "noninterest_expense": M(FLOW, ["NoninterestExpense"], quarterly=True),
    "provision_for_losses": M(FLOW, [
        "ProvisionForLoanAndLeaseLosses", "ProvisionForLoanLeaseAndOtherLosses",
        "ProvisionForCreditLosses",
    ], quarterly=True),

    # Loan-loss realization (for normalized loss rate). Tag names vary across
    # three eras — pre-CECL "AllowanceForLoanAndLeaseLosses…", the interim
    # "FinancingReceivableAllowanceForCreditLosses…", and the current
    # "FinancingReceivableExcludingAccruedInterest…" — so each list spans all
    # three to get continuous history (extract_annual_series merges years
    # across the list). Net charge-offs are taken directly from the
    # "write-off after recovery" tags where present, else derived in app.py
    # from gross write-offs − recoveries.
    "net_charge_offs_reported": M(FLOW, [
        "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery",
        "FinancingReceivableAllowanceForCreditLossWriteoffAfterRecovery",
        "AllowanceForLoanAndLeaseLossesWriteoffsNet",
    ], row=False, note="Feeds net_charge_offs, which is the row."),
    "loan_writeoffs": M(FLOW, [
        "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoff",
        "FinancingReceivableAllowanceForCreditLossesWriteOffs",
        "AllowanceForLoanAndLeaseLossesWriteOffs",
    ], row=False, note="Gross half of the derived net charge-offs row."),
    "loan_recoveries": M(FLOW, [
        "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossRecovery",
        "FinancingReceivableAllowanceForCreditLossesRecovery",
        "AllowanceForLoanAndLeaseLossRecoveryOfBadDebts",
    ], row=False, note="Recovery half of the derived net charge-offs row."),

    # ── Per share (USD/shares unit) ─────────────────────────────────────────
    "eps_diluted": M(FLOW, ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"]),

    # Declared first, cash-paid second. Declared is the rate the board set for
    # the year; cash-paid shifts a quarter whenever a December declaration
    # settles in January, which understates a payer in the year its timing
    # moved. Monthly payers aggregate correctly either way — Realty Income's
    # CY2025 frame is $3.217, all twelve declarations, not one of them.
    #
    # Both spellings are needed, and which one a filer uses is not predictable.
    # Declared covers Procter & Gamble, Verizon, AT&T, Apple, Microsoft,
    # JPMorgan and Brandywine; CashPaid covers Coca-Cola, Johnson & Johnson,
    # Realty Income, Simon, Exxon and Main Street, none of which tag Declared
    # at all in recent periods (Exxon has never tagged it — the concept 404s
    # for that filer). Merging the two gives 1,650 filers a current quarterly
    # rate; Declared alone gives 1,343 and drops every name in that second list.
    "dividends_per_share": M(FLOW, [
        "CommonStockDividendsPerShareDeclared",
        "CommonStockDividendsPerShareCashPaid",
    ], quarterly=True),

    # ── Cash flow ───────────────────────────────────────────────────────────
    "operating_cash_flow": M(FLOW, [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",  # DPZ 2014-2016 and similar
        IFRSC("CashFlowsFromUsedInOperatingActivities"),
    ], quarterly=True),

    # The first two are the only ones the screen reads. The rest are
    # industry-specific spellings that pay off when you are looking at one
    # filer's fifteen years and would widen the screen's capex definition
    # unevenly across sectors if applied to a whole `frames` slice.
    "capex": M(FLOW, [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        A("PaymentsToAcquireOtherPropertyPlantAndEquipment"),
        A("PaymentsToAcquireMachineryAndEquipment"),       # Seadrill (SDRL) and similar
        A("PaymentsToAcquireOilAndGasPropertyAndEquipment"),
        A("PaymentsToExploreAndDevelopOilAndGasProperties"),  # E&P drilling capex (CHRD, APA, and similar)
        A("PaymentsToAcquirePropertyPlantEquipmentAndIntangibleAssets"),
        A("SegmentExpenditureAdditionToLongLivedAssets"),
        A("PaymentsForCapitalImprovements"),               # Noble Corporation (NE) and similar
        IFRSC("PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"),
    ], quarterly=True),

    "depreciation": M(FLOW, [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        A("Depreciation"),
        A("CostOfGoodsAndServicesSoldDepreciationAndAmortization"),  # NE (Noble) and similar drillers
    ], quarterly=True),

    "stock_based_compensation": M(FLOW, [
        "ShareBasedCompensation", "AllocatedShareBasedCompensationExpense",
        "StockBasedCompensation", "EmployeeBenefitsAndShareBasedCompensation",
    ], quarterly=True),

    "intangible_amortization": M(FLOW, [
        "AmortizationOfIntangibleAssets",
        "AmortizationOfIntangibleAssetsExcludingGoodwill",
    ], row=False, note="Add-back inside the UNTA and NOPAT derivations."),

    # Unrealized investment gains / (losses) only — positive = gain, negative = loss.
    # Realized gains are kept in net income (they represent actual cash transactions).
    # Unrealized gains are stripped out because they are pure mark-to-market noise
    # that distorts recurring earning power (especially post-ASC 321, 2018+).
    "investment_gains": M(FLOW, [
        "UnrealizedGainLossOnInvestments",
        "EquitySecuritiesFvNiUnrealizedGainLoss",         # most common post-ASC 321 tag
        "TradingSecuritiesUnrealizedHoldingGainLoss",
        "UnrealizedGainLossOnSecurities",
    ], quarterly=True, row=False,
        note="Stripped out of normalized earnings rather than shown on its own."),

    # ── Balance sheet — point in time ───────────────────────────────────────
    "total_assets": M(INSTANT, ["Assets"], quarterly=True),

    # Operating leases (ASC 842, on-balance-sheet from 2019). Kept as separate
    # component metrics rather than one list, because OperatingLeaseLiability is
    # the TOTAL while Current/Noncurrent are its parts — merging them into one
    # candidate list would silently pick whichever is larger. The true total is
    # derived in app.py.
    "operating_lease_liability_total": M(INSTANT, ["OperatingLeaseLiability"],
        quarterly=True, row=False, note="Feeds the derived operating_lease_liability row."),
    "operating_lease_liability_current": M(INSTANT, ["OperatingLeaseLiabilityCurrent"],
        quarterly=True, row=False, note="Feeds the derived operating_lease_liability row."),
    "operating_lease_liability_noncurrent": M(INSTANT, ["OperatingLeaseLiabilityNoncurrent"],
        quarterly=True, row=False, note="Feeds the derived operating_lease_liability row."),
    "operating_lease_rou_asset": M(INSTANT, ["OperatingLeaseRightOfUseAsset"], quarterly=True),

    # Loans held (denominator for net charge-off rate). Tag varies by era:
    # net-of-allowance pre-CECL, then the CECL-era financing-receivable tag.
    "bank_loans": M(INSTANT, [
        "LoansAndLeasesReceivableNetReportedAmount",
        "LoansAndLeasesReceivableNetOfDeferredIncome",
        "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
        "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
    ], row=False, note="Denominator of the NCO rate row; not a line of its own."),

    "current_assets":      M(INSTANT, ["AssetsCurrent"], quarterly=True, row=False,
        note="Feeds the working capital row."),
    "current_liabilities": M(INSTANT, ["LiabilitiesCurrent"], quarterly=True, row=False,
        note="Feeds the working capital row."),
    "total_liabilities":   M(INSTANT, ["Liabilities"], quarterly=True),

    "equity": M(INSTANT, [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        # Limited partnerships / MLPs report partners' capital, not stockholders'
        # equity (e.g. PAGP – Plains GP Holdings LP). Plain (parent) tag first,
        # then the including-NCI total, then LLC members' equity.
        A("PartnersCapital"),
        A("PartnersCapitalIncludingPortionAttributableToNoncontrollingInterest"),
        A("MembersEquity"),
        A("MembersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        IFRSC("Equity"),
    ], quarterly=True),

    "cash": M(INSTANT, [
        "CashAndCashEquivalentsAtCarryingValue",
        A("CashCashEquivalentsAndShortTermInvestments"),
        # ASC 230 / post-2017 standard tag; also used by BRK after 2017
        A("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        IFRSC("CashAndCashEquivalents"),
    ], quarterly=True),

    "short_term_investments": M(INSTANT, [
        "ShortTermInvestments", "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "AvailableForSaleSecuritiesCurrent",
        "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent",  # INTU post-FY2023
    ], quarterly=True),

    # Marketable equity portfolio. Only surfaced as a row where a company
    # template asks for it (Berkshire), but extracted generally so the QUARTERLY
    # value exists for any filer whose UNTA binding deducts it — an annual UNTA
    # that nets out the portfolio beside a quarterly one that doesn't would be
    # two different definitions in one row. ASU 2016-01 moved these off
    # available-for-sale accounting, hence both concepts.
    #
    # kind=INSTANT resolves a disagreement between the two extraction paths in
    # app.py: the quarterly pass listed this among its point-in-time metrics and
    # the annual pass did not, so the same balance-sheet position was read
    # first-tag-wins in one column and largest-absolute-value in the others.
    # The two rules agree here — the concepts belong to non-overlapping eras
    # (AvailableForSale… before ASU 2016-01, EquitySecuritiesFvNi after) and a
    # filer in the transition year reports the same figure under both — so the
    # annual column is unchanged by being classified correctly.
    "equity_securities": M(INSTANT, [
        "EquitySecuritiesFvNi",
        "AvailableForSaleSecuritiesEquitySecurities",
    ], quarterly=True, row=False,
        note="Deducted inside UNTA; surfaced as a row only by company template."),

    # Consolidated debt, in preference order — first that resolves wins.
    #
    # Bare LongTermDebt is LAST among the totals on purpose: it is the generic
    # concept and filers use it for wildly different scopes, sometimes a
    # fraction of the real balance (DPZ tags ~$14M of a securitized structure
    # worth billions). One filer reads $23M under it against $15.3B under
    # LongTermDebtAndFinanceLeaseObligations.
    #
    # The screener used to put it second and so read the partial figure for 107
    # of 3,682 filers; unifying on this order moved those and left the other
    # 3,575 untouched.
    "long_term_debt": M(INSTANT, [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",       # DPZ securitization + leases (noncurrent)
        "LongTermDebtAndFinanceLeaseObligations",
        A("DebtAndFinanceLeaseObligationsNoncurrent"),
        A("LongTermNotesPayable"),                      # ORCL (annual 10-K noncurrent)
        A("LongTermNotesAndLoans"),                     # ORCL (quarterly 10-Q noncurrent)
        # Screen-only, and this one is not drift to be closed. Both surfaces
        # build total debt as long_term_debt + current_debt, and this concept
        # already contains both, so on a surface that adds the current portion
        # separately it double counts. The screen has carried it since before
        # that was understood; adding it to the analyze page would import the
        # fault rather than share a vocabulary.
        S("DebtLongtermAndShorttermCombinedAmount"),
        "LongTermDebt",                                 # generic, may be partial (e.g. DPZ ~$14M only)
        A("FinanceLeaseLiabilityNoncurrent"),           # last: lease-financed cos (e.g. LIVE post-2022 sale-leasebacks)
        IFRSC("NoncurrentPortionOfNoncurrentBorrowings"),
        IFRSC("LongtermBorrowings"),

        # Components, summed by the screen only when no total resolved, and
        # gap-filling the analyze page's debt row per period for the same
        # reason. A REIT does not file a classified balance sheet, so
        # "noncurrent" rarely applies and many tag no consolidated total at
        # all: Realty Income prints revolving credit and commercial paper, term
        # loans, mortgages payable and notes payable as four separate lines
        # totalling ~$28.8B, none of them us-gaap:LongTermDebt. Binding only
        # the totals resolved NOTHING for it, and enterprise value silently
        # collapsed to market cap less cash — which is what put Empire State
        # Realty at a 46% implied cap rate.
        #
        # One concept per debt family, so two names for the same facility
        # cannot both land; and a total always wins over the sum, so a filer
        # tagging both is not double counted.
        #
        # On the analyze side only senior notes gap-fill, and only where the
        # whole totals list came back empty: VeriSign's sole borrowing is a
        # note issue tagged SeniorLongTermNotes / SeniorNotes ($1,788,200
        # thousand at FY2025) and none of the totals, so its debt, net cash and
        # enterprise value were blank for fifteen years. Widening the analyze
        # gap-fill to the other components would risk letting a component stand
        # in for a total, or pushing a figure that already includes the current
        # portion into a row current debt is then added to.
        #
        # DebtInstrumentFairValue is deliberately not used. VeriSign carries it
        # at $1,750,000 thousand against the $1,788,200 thousand carrying
        # amount, and a fair value is a different measure, not a substitute.
        Concept("SeniorLongTermNotes", GAAP, {ANALYZE: GAP_FILL}),
        P("SecuredDebt"),
        P("UnsecuredDebt"),
        Concept("SeniorNotes", GAAP, {ANALYZE: GAP_FILL, SCREEN: COMPONENT}),
        P("NotesPayable"),
        P("LongTermLineOfCredit"),
        P("CommercialPaper"),
        P("LoansPayable"),
    ], quarterly=True),

    "current_debt": M(INSTANT, [
        A("LongTermDebtAndCapitalLeaseObligationsCurrent"),  # DPZ current portion of securitization
        "LongTermDebtCurrent",
        "DebtCurrent",
        A("NotesPayableCurrent"),                       # ORCL quarterly current debt
        "ShortTermBorrowings",
        A("CurrentPortionOfLongTermDebt"),
        A("LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"),
        A("CommercialPaper"),
        A("ShortTermDebt"),
        A("FinanceLeaseLiabilityCurrent"),              # last: lease-financed cos (e.g. LIVE)
        IFRSC("CurrentPortionOfNoncurrentBorrowings"),
        IFRSC("ShorttermBorrowings"),
    ], quarterly=True),

    "goodwill":    M(INSTANT, ["Goodwill"], quarterly=True),
    "intangibles": M(INSTANT, ["FiniteLivedIntangibleAssetsNet",
                               "IntangibleAssetsNetExcludingGoodwill"], quarterly=True),
    "inventory":   M(INSTANT, ["InventoryNet"], quarterly=True, row=False,
        note="Feeds the working capital row."),

    # Net productive fixed assets. The single most widespread gap in the whole
    # mapping before this existed: the coverage audit flagged it for 91 of 118
    # S&P 500 names, at up to 85% of total assets. For an asset-heavy filer it
    # is the asset the business actually runs on, and nothing consumed it.
    #
    # The second tag is the same concept under its post-ASC-842 caption, which
    # folds finance-lease right-of-use assets into the line. Companies switch
    # cleanly: T, LUMN, UNP and CVS all report BOTH in their transition year
    # and the two agree to 0.00% ($130.13B for T in 2019, $57.40B for UNP in
    # 2023), so first-tag-wins joins the eras without a step.
    #
    # Deliberately NOT including RealEstateInvestmentPropertyNet: that is a
    # REIT's operating property and already has its own `real_estate_assets`
    # row. Folding it in here would put two different concepts in one row and,
    # for a filer reporting both (Prologis reports $0.2B of corporate PP&E
    # beside $80B of real estate), would splice the series mid-history.
    "ppe_net": M(INSTANT, [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ], quarterly=True),

    # ── Shares ──────────────────────────────────────────────────────────────
    # The analyze page wants the balance-sheet count dated to the period it is
    # showing, so the us-gaap concept leads. The screen wants the most recent
    # count known for a live market cap, and `_cf_extract` gets there by its own
    # recency rule rather than by tag order — a cover-page count is dated later
    # than the period end, so the dei concept wins there regardless of position.
    "shares_outstanding_end": M(INSTANT, [
        "CommonStockSharesOutstanding",
        DEIC("EntityCommonStockSharesOutstanding"),
        IFRSC("NumberOfSharesIssued"),
        # 20-F filers often lack the dei cover-page share count in
        # companyfacts, and some filers (META) tag no share count at all. The
        # FY weighted average sizes a market cap well enough for the screen and
        # fills a quarter column well enough to divide by; it is not a
        # period-end count and nothing applies it to the annual column.
        Concept("WeightedAverageNumberOfSharesOutstandingBasic", GAAP,
                {ANALYZE: PROXY, SCREEN: PROXY}),
        IFRSC("WeightedAverageShares", PROXY),
    ], quarterly=True),

    "shares_diluted_wtd": M(FLOW, [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ], row=False, note="Denominator of every per-share row; not a line of its own."),

    # ── Capital returns ─────────────────────────────────────────────────────
    # PaymentsOfOrdinaryDividends covers filers (e.g. ACN from FY2023) that
    # switched away from PaymentsOfDividendsCommonStock. Excludes
    # PaymentsOfDividendsMinorityInterest (dividends to subsidiary minority
    # holders, not the company's own shareholders).
    "dividends_paid": M(FLOW, [
        "PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
        "PaymentsOfOrdinaryDividends",
    ], quarterly=True),

    "buybacks_value": M(FLOW, [
        "PaymentsForRepurchaseOfCommonStock",
        "StockRepurchasedAndRetiredDuringPeriodValue",
        "StockRepurchasedDuringPeriodValue",
    ], quarterly=True),

    "treasury_stock": M(INSTANT, ["TreasuryStockCommonValue", "TreasuryStockValue"],
        quarterly=True, row=False, note="Deducted inside book value; not a line of its own."),

    "shares_repurchased": M(FLOW, [
        "StockRepurchasedAndRetiredDuringPeriodShares",
        "StockRepurchasedDuringPeriodShares",
        "TreasuryStockSharesAcquired",
    ]),

    "treasury_stock_shares": M(INSTANT, ["TreasuryStockCommonShares", "TreasuryStockShares"],
        quarterly=True, row=False, note="Reconciles issued to outstanding shares; not a line of its own."),

    # Buyback program remaining (best-effort; not all companies report via XBRL)
    "buyback_remaining": M(INSTANT, [
        "StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1",
        "StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount",
    ], row=False, note="Shown in the buyback summary panel, not the statement table."),

    # ── REIT-specific metrics ───────────────────────────────────────────────
    # Gains / losses on real estate dispositions — subtracted in FFO derivation
    "gains_on_real_estate": M(FLOW, [
        "GainLossOnSaleOfProperties",
        "GainsLossesOnSalesOfInvestmentRealEstate",
        "GainLossOnDispositionOfRealEstateAssets",
        "GainOnSaleOfProperties",
        "GainLossOnSaleOfPropertiesBeforeApplicableIncomeTaxes",  # SPG and similar
    ], quarterly=True, row=False, note="Backed out inside the FFO row."),

    # Real property depreciation — added back in FFO (may differ from total D&A).
    # Only REIT-specific tags here; general D&A fallback is applied in
    # build_financials but only when other REIT signals (real_estate_assets,
    # straight_line_rent) are present.
    "real_estate_depreciation": M(FLOW, [
        "DepreciationOfRealEstate",          # standard REIT tag (most REITs file this)
        "RealEstateDepreciationAndAmortization",
    ], quarterly=True, row=False, note="Added back inside the FFO row."),

    # Straight-line rent adjustment — stripped out in AFFO derivation
    "straight_line_rent": M(FLOW, ["StraightLineRent", "StraightLineRentAdjustments"],
        quarterly=True, row=False, note="Stripped out inside the AFFO row."),

    # Net real estate assets on balance sheet
    "real_estate_assets": M(INSTANT, [
        "RealEstateInvestmentPropertyNet",
        "RealEstateAndAccumulatedDepreciation",  # alternative tag
    ], quarterly=True),

    # Recurring (maintenance/tenant improvement) capex — used in AFFO
    "recurring_capex": M(FLOW, ["PaymentsForTenantImprovements", "PaymentsForLeasingCosts"],
        quarterly=True, row=False, note="Deducted inside the AFFO row."),

    # ── BDC / investment company metrics ────────────────────────────────────
    # Net Investment Income (flow – income statement equivalent for BDCs)
    "net_investment_income": M(FLOW, [
        "NetInvestmentIncome",
        "InvestmentIncomeOperatingAfterExpenseAndTax",
    ], quarterly=True),

    # Gross Investment Income (top-line "revenue" equivalent for BDCs)
    "gross_investment_income": M(FLOW, [
        "GrossInvestmentIncomeOperating",
        "InvestmentIncomeInterestAndDividend",
        "InvestmentIncomeInterest",
    ], quarterly=True),

    # NAV per share (point-in-time, filed directly in XBRL)
    "nav_per_share": M(INSTANT, ["NetAssetValuePerShare"], quarterly=True),

    # NII per share (flow – per-share NII as reported in financial highlights)
    "nii_per_share": M(FLOW, [
        "InvestmentCompanyInvestmentIncomeLossPerShare",
        "InvestmentCompanyInvestmentIncomeLossFromOperationsPerShare",
    ], quarterly=True),

    # ── Insurance (P&C / life) metrics ──────────────────────────────────────
    # Net premiums earned — the insurer's top-line "revenue" for ratio math.
    # As with non-interest expense, the screen reads only the first as a
    # membership test: a filer that tags net premiums earned is an insurer.
    "premiums_earned": M(FLOW, [
        "PremiumsEarnedNet",
        A("PremiumsEarnedNetPropertyAndCasualty"),
        A("SupplementaryInsuranceInformationPremiumRevenue"),
    ], quarterly=True),

    # Net premiums written (leading indicator of earned premium growth)
    "premiums_written": M(FLOW, [
        "PremiumsWrittenNet",
        "SupplementaryInsuranceInformationPremiumsWritten",
    ], quarterly=True),

    # Losses & loss-adjustment expenses incurred (numerator of the loss ratio)
    "losses_incurred": M(FLOW, [
        "PolicyholderBenefitsAndClaimsIncurredNet",
        "LiabilityForUnpaidClaimsAndClaimsAdjustmentExpenseIncurredClaims1",
        "SupplementaryInsuranceInformationBenefitsClaimsLossesAndSettlementExpense",
    ], quarterly=True),

    # Total benefits, losses & expenses (numerator of the combined ratio)
    "benefits_losses_expenses": M(FLOW, ["BenefitsLossesAndExpenses"],
        quarterly=True, row=False, note="Numerator of the combined ratio row."),

    # Loss & LAE reserves (largest float component). The screen reads only the
    # first: with unearned premiums it is the float proxy that scales to
    # hundreds of tickers, coarser than a company-specific extraction (this
    # app's own BRK build uses the as-reported float instead).
    "claims_reserve": M(INSTANT, [
        "LiabilityForClaimsAndClaimsAdjustmentExpense",
        A("LiabilityForUnpaidClaimsAndClaimsAdjustmentExpenseNet"),
        A("SupplementaryInsuranceInformationLiabilityForFuturePolicyBenefitsLossesClaimsAndLossExpenseReserves"),
    ], quarterly=True, row=False, note="Largest component of the insurance float row."),

    # Unearned premium reserve (float component).
    #
    # UnearnedPremiumsLiability is the concept the screen reads and the analyze
    # page did not have at all — the audit found it as drift. It is a gap-fill
    # here rather than a series entry: where a filer tags both this and
    # UnearnedPremiums the two are the same reserve under two captions, and
    # first-tag-wins should keep the one already in use.
    "unearned_premiums": M(INSTANT, [
        A("UnearnedPremiums"),
        A("SupplementaryInsuranceInformationUnearnedPremiums"),
        Concept("UnearnedPremiumsLiability", GAAP, {ANALYZE: GAP_FILL, SCREEN: SERIES}),
    ], quarterly=True, row=False, note="Component of the insurance float row."),

    # Premiums receivable (offsets float — money not yet collected)
    "premiums_receivable": M(INSTANT, ["PremiumsReceivableAtCarryingValue"],
        quarterly=True, row=False, note="Offsets the insurance float row."),

    # Deferred policy acquisition costs (offsets float)
    "deferred_acquisition_costs": M(INSTANT, [
        "DeferredPolicyAcquisitionCosts",
        "SupplementaryInsuranceInformationDeferredPolicyAcquisitionCosts",
    ], quarterly=True, row=False, note="Offsets the insurance float row."),

    # Reinsurance recoverables (offsets float)
    "reinsurance_recoverable": M(INSTANT, [
        "ReinsuranceRecoverablesOnPaidAndUnpaidLosses",
        "ReinsuranceRecoverableForUnpaidClaimsAndClaimsAdjustments",
    ], quarterly=True, row=False, note="Offsets the insurance float row."),

    # ── Introduced by company templates ─────────────────────────────────────
    # These carry no global concepts: nothing here is read for a filer whose
    # template does not ask for it, which is the whole point of the mechanism —
    # a coverage gap that is real for one company is noise for the next.
    #
    # What the vocabulary owns even so is what the metric IS: whether it is a
    # balance or a flow, and whether it is a row. A template used to declare its
    # own `point_in_time` list, which was a second copy of `kind` living in a
    # JSON file, and it was already wrong in the same way the app's two copies
    # were: the annual pass merged it and the quarterly pass never did, so a
    # template-introduced balance would have been read largest-value in the
    # quarter columns. Inert only because none of them is a quarterly metric.
    #
    # All three are inputs to ratios a template defines, not rows themselves —
    # DXC divides backlog by revenue, DXC and LUMN divide cumulative impairment
    # by total assets — so the template renders the ratio and the raw balance
    # stays behind it.
    "goodwill_impairment_accumulated": M(INSTANT, [], by_template=True, row=False,
        note="Numerator of a company-template ratio (DXC, LUMN); the ratio is the row."),
    "backlog_rpo": M(INSTANT, [], by_template=True, row=False,
        note="Numerator of a company-template ratio (DXC); the ratio is the row."),
    "real_estate_gross": M(INSTANT, [], by_template=True, row=False,
        note="Real estate before depreciation (REIT sector template, SPG); "
             "feeds template rows, not a row of its own."),
}


# ═════════════════════════════════════════════════════════════════════════════
# Accessors — the only way a consumer should reach a concept name
# ═════════════════════════════════════════════════════════════════════════════

def tags(metric: str, roles=SERIES, surface: str = ANALYZE, ns=(GAAP, DEI)) -> list[str]:
    """Element names for one metric, in vocabulary order.

    `roles` may be a single role or a tuple; a tuple returns them interleaved in
    vocabulary order, which is what a first-hit-wins consumer wants (series
    entries first, gap-fills as the tail).
    """
    roles = (roles,) if isinstance(roles, str) else tuple(roles)
    ns    = (ns,) if isinstance(ns, str) else tuple(ns)
    m = VOCAB.get(metric)
    if not m:
        return []
    return [c.tag for c in m.concepts
            if c.ns in ns and c.role(surface) in roles]


def ns_tags(metric: str, roles=SERIES, surface: str = SCREEN) -> list[tuple[str, str]]:
    """(namespace, element) pairs — for consumers that query more than us-gaap."""
    roles = (roles,) if isinstance(roles, str) else tuple(roles)
    m = VOCAB.get(metric)
    if not m:
        return []
    return [(c.ns, c.tag) for c in m.concepts if c.role(surface) in roles]


def metric_tags(surface: str = ANALYZE) -> dict[str, list[str]]:
    """{metric: priority-ordered element names} — the shape app.py's
    METRIC_TAGS has always had, now derived rather than declared."""
    return {k: tags(k, SERIES, surface) for k in VOCAB}


def gap_fill_tags(surface: str = ANALYZE) -> dict[str, list[str]]:
    """{metric: element names applied only where the series list resolved
    nothing}. Metrics with no gap-fill are omitted."""
    out = {}
    for k in VOCAB:
        t = tags(k, GAP_FILL, surface)
        if t:
            out[k] = t
    return out


def instants() -> set[str]:
    return {k for k, m in VOCAB.items() if m.kind == INSTANT}


def quarterly(kind=None) -> set[str]:
    return {k for k, m in VOCAB.items()
            if m.quarterly and (kind is None or m.kind == kind)}


def rendered() -> set[str]:
    """Metrics that must have a UI row. A metric that resolves but is not shown
    reads as 'no such measure' rather than 'missing', which is the trap
    pretax_income fell into — so the absence of a row is a declaration here
    (`row=False` plus a note), not an omission in the template."""
    return {k for k, m in VOCAB.items() if m.row}


def internal() -> set[str]:
    return {k for k, m in VOCAB.items() if not m.row}


def by_template() -> set[str]:
    """Metrics whose concepts come per filer from a company template."""
    return {k for k, m in VOCAB.items() if m.by_template}


def template_conflicts(metric: str, added: list) -> list:
    """[(tag, why)] for concepts a company template must NOT add to `metric`.

    A template extends the vocabulary for one filer, which is the mechanism
    that lets a coverage gap real for one company stay noise for the next. What
    it must not do is reach past a decision the vocabulary made about the
    analyze surface itself, because that decision is what keeps a row meaning
    one thing.

    The case this was written for: ORCL's template appended
    DebtLongtermAndShorttermCombinedAmount and UnsecuredDebt to long_term_debt.
    The first is declared here as screen-only because it already contains the
    current portion, and the analyze page adds current_debt to this row
    separately — so it double counts on exactly the surface a template extends.
    The second is a debt COMPONENT, which may stand for a total only where no
    total resolved, never alongside one. Neither ever won for ORCL (an earlier
    concept resolved in all 18 annual years and all 3 quarters), so the entry
    was dead weight rather than a live fault — but it was one filing away from
    being live, and nothing would have said so.

    An unknown concept is fine and expected: that IS the extension.
    """
    m = VOCAB.get(metric)
    if not m:
        return [(t, f"no metric named {metric!r} in the vocabulary") for t in added]
    out = []
    declared = {c.tag: c for c in m.concepts}
    for tag in added:
        c = declared.get(tag)
        if c is None:
            continue                       # a genuine per-filer extension
        role = c.role(ANALYZE)
        if role is None:
            out.append((tag, f"declared for {metric} on the screen only — "
                             f"the vocabulary keeps it off the analyze surface"))
        elif role != SERIES:
            out.append((tag, f"declared for {metric} as a {role}, which is not "
                             f"interchangeable with the candidate list"))
    return out
