import math

import pytest
from pydantic import ValidationError

from nyx.contract import Question, Request, answer, candidates


def test_choice_contract_and_answer():
    question = Question(type="choice", instructions="Pick one", criteria={"a": None, "b": "second"})
    assert candidates(question)[0] == ["a", "b"]
    result = answer(question, [0.25, 0.75])
    assert result["choice"] == "b"
    assert result["probabilities"] == {"a": 0.25, "b": 0.75}


def test_noul_returns_true_probability():
    question = Question(type="noul", instructions="Is it true?", criteria=None)
    assert answer(question, [0.1, 0.9]) == {"type": "noul", "noul": 0.9}


def test_request_rejects_unknown_fields_and_empty_questions():
    with pytest.raises(ValidationError):
        Request(model="nyx", state="x", questions={}, extra=True)


@pytest.mark.parametrize("values", [[0, 0], [math.nan, 1], [-1, 2]])
def test_answer_rejects_invalid_probabilities(values):
    question = Question(type="noul", instructions="x", criteria=None)
    with pytest.raises(ValueError, match="Invalid model probabilities"):
        answer(question, values)
