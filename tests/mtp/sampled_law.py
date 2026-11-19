"""Exact finite-vocabulary checks for the native sampled MTP contract."""

from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import prod


EOS = 2
ROWS = (
    (Fraction(1, 2), Fraction(3, 8), Fraction(1, 8)),
    (Fraction(0), Fraction(1, 4), Fraction(3, 4)),
    (Fraction(1, 4), Fraction(3, 4), Fraction(0)),
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(1, 8), Fraction(3, 8), Fraction(1, 2)),
)


def target(history):
    return ROWS[(len(history) + sum((index + 1) * token
                                   for index, token in enumerate(history))) % len(ROWS)]


def proposal(history, mode):
    row = target(history)
    if mode == "argmax":
        return tuple(Fraction(token == row.index(max(row))) for token in range(len(row)))
    if mode == "target":
        return row
    if mode == "outside":
        return tuple(Fraction(token == row.index(min(row))) for token in range(len(row)))
    if mode == "skewed":
        return (Fraction(1, 8), Fraction(3, 4), Fraction(1, 8))
    return tuple(reversed(row))


def sample_law(row):
    return {token: weight for token, weight in enumerate(row) if weight}


def finished(history, budget, stops):
    return bool(history and (history[-1] in stops or len(history) == budget))


def target_sequences(budget, stops):
    outputs = defaultdict(Fraction)

    def visit(history, weight):
        if finished(history, budget, stops):
            outputs[history] += weight
            return
        for token, probability in sample_law(target(history)).items():
            visit((*history, token), weight * probability)

    visit((), Fraction(1))
    return dict(outputs)


def accepted_round(history, draft, offset, budget, stops):
    row = target(history)
    for token, probability in sample_law(row).items():
        continuation = (*history, token)
        match = offset < len(draft) and token == draft[offset]
        if not match or finished(continuation, budget, stops):
            yield continuation, probability
        else:
            for output, suffix in accepted_round(continuation, draft, offset + 1, budget, stops):
                yield output, probability * suffix


def speculative_sequences(depth, mode, budget, stops):
    outputs = defaultdict(Fraction)

    def visit(history, weight):
        if finished(history, budget, stops):
            outputs[history] += weight
            return
        slots = min(depth, budget - len(history) - 1) if history else 0
        for draft in product(range(len(ROWS[0])), repeat=slots):
            draft_weight = prod(proposal(history + draft[:index], mode)[token]
                                for index, token in enumerate(draft))
            if draft_weight:
                for output, probability in accepted_round(history, draft, 0, budget, stops):
                    visit(output, weight * draft_weight * probability)

    visit((), Fraction(1))
    return dict(outputs)


def draw(history, ordinal, coins):
    total = Fraction()
    for token, probability in sample_law(target(history)).items():
        total += probability
        if coins[ordinal] < total:
            return token
    raise AssertionError("Coin must be in [0, 1)")


def draw_chain(coins, stops, depth, mode):
    committed = ()
    while not finished(committed, len(coins), stops):
        slots = min(depth, len(coins) - len(committed) - 1) if committed else 0
        draft = ()
        for _ in range(slots):
            row = proposal(committed + draft, mode)
            draft += (row.index(max(row)),)
        draws = tuple(draw(committed + draft[:index], len(committed) + index, coins)
                      for index in range(len(draft) + 1))
        for index, token in enumerate(draws):
            committed += (token,)
            if (finished(committed, len(coins), stops) or index == len(draft)
                    or token != draft[index]):
                break
    return committed


def check_local_laws():
    for row in ROWS:
        for token, mass in enumerate(row):
            via_residual = tuple(
                mass if other == token else (1 - mass) * weight / (1 - mass)
                if mass < 1 else Fraction()
                for other, weight in enumerate(row)
            )
            assert via_residual == row, (row, token)
    target_row = ROWS[0]
    draft_row = (Fraction(1, 8), Fraction(3, 4), Fraction(1, 8))
    equality = sum(p * q for p, q in zip(target_row, draft_row, strict=True))
    ratio_joint = tuple(min(p, q) for p, q in zip(target_row, draft_row, strict=True))
    residual = tuple(p - accepted for p, accepted in zip(target_row, ratio_joint, strict=True))
    assert equality == Fraction(23, 64)
    assert sum(ratio_joint) == Fraction(5, 8)
    assert tuple(accepted + correction
                 for accepted, correction in zip(ratio_joint, residual, strict=True)) == target_row


def main():
    modes = ("argmax", "target", "outside", "skewed", "reversed")
    stop_policies = ((), (EOS,), (0, EOS))
    check_local_laws()
    laws = 0
    for budget, stops in product(range(1, 7), stop_policies):
        wanted = target_sequences(budget, stops)
        assert sum(wanted.values()) == 1
        for depth, mode in product(range(1, 5), modes):
            got = speculative_sequences(depth, mode, budget, stops)
            assert got == wanted, (budget, stops, depth, mode)
            laws += 1
    coupled = 0
    grid = tuple(Fraction(value, 8) for value in (0, 3, 4, 6, 7))
    for coins, stops in product(product(grid, repeat=5), stop_policies):
        wanted = draw_chain(coins, stops, 0, "argmax")
        for depth, mode in product(range(1, 5), modes):
            assert draw_chain(coins, stops, depth, mode) == wanted, (coins, stops, depth, mode)
            coupled += 1
    print(f"{laws} exact sequence laws and {coupled} output-index coupling cases passed")


if __name__ == "__main__":
    main()
