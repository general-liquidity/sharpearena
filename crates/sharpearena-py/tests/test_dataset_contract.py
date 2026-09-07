"""Prompt examples must execute under the parser, without installing a model."""

import json
import re

import numpy as np
import pytest

from sharpearena.dataset import _initial_question
from sharpearena.decision_examples import render_decision_examples
from sharpearena.decision_parser import parse_decision, parse_decision_payload
from sharpearena.verifiers_env import render_observation


def _check_examples(question, symbols):
    examples = re.findall(r"<action>(.*?)</action>", question)
    assert examples, "a prompt with no examples must not pass vacuously"
    assert question.count("<action>") == question.count("</action>") == 3
    current = np.arange(1, len(symbols) + 1, dtype=float) / 10
    # The parser also accepts XML wrappers. Requiring bare JSON here catches
    # an unmatched opening tag that its second extraction could otherwise hide.
    raw_payloads = [json.loads(example) for example in examples]
    payloads = [parse_decision_payload(example) for example in examples]
    assert raw_payloads == payloads
    targets = [
        parse_decision(example, symbols, current_weights=current)
        for example in examples
    ]
    assert len(examples) == 3
    assert targets[0][0] == 0.25
    assert payloads[1]["orders"] == []
    np.testing.assert_array_equal(targets[1], current)  # hold is not flatten
    assert all(order["action"] == "close" for order in payloads[2]["orders"])
    assert {order["symbol"] for order in payloads[2]["orders"]} == set(symbols)
    np.testing.assert_array_equal(targets[2], np.zeros(len(symbols)))
    if len(symbols) > 1:
        np.testing.assert_array_equal(targets[0][1:], current[1:])
    assert "omitted symbols retain" in question.lower()


@pytest.mark.parametrize("n_symbols", [1, 2, 4])
@pytest.mark.parametrize("allow_short", [True, False])
def test_initial_prompt_teaches_accepted_actions_and_sparse_hold(n_symbols, allow_short):
    question = _initial_question(n_symbols, 20, "train", allow_short, "test mandate")
    _check_examples(question, [f"SYM{i:02}" for i in range(n_symbols)])
    assert ("[-1, 1]" if allow_short else "[0, 1]") in question


def test_turn_prompt_uses_the_observed_symbol_names():
    symbols = ["AAA", "BTC-USDT", "third"]
    obs = {"closes": [100, 200, 300], "positions": [0, 0, 0], "cash": [1000]}
    _check_examples(render_observation(obs, symbols), symbols)
    assert "<action>" not in render_observation(obs, symbols, final=True)


@pytest.mark.parametrize("symbols", [[], [""], [None], ["AAA", "AAA"]])
def test_example_generation_rejects_invalid_symbol_axes(symbols):
    with pytest.raises(ValueError, match="decision examples require"):
        render_decision_examples(symbols)


def test_xml_delimiters_in_symbols_do_not_break_example_envelopes():
    symbols = ['odd</action><action>"&', "normal"]
    _check_examples(render_decision_examples(symbols), symbols)


def test_examples_validate_against_the_published_native_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from sharpearena import decision_schema_json

    schema = json.loads(decision_schema_json())
    examples = re.findall(r"<action>(.*?)</action>", render_decision_examples(["AAA", "BBB"]))
    assert len(examples) == 3
    for example in examples:
        jsonschema.validate(json.loads(example), schema)
