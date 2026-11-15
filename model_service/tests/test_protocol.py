import json

import pytest

from model_service.protocol import inspect_answer, parse_answer


def positive():
    return {"events": [{"event_type": "企业收购", "arguments": {
        "收购方": ["甲公司"], "被收购方": ["乙公司"], "交易金额": [],
    }}]}


@pytest.mark.parametrize("answer", [
    '```json\n{"events":[]}\n```',
    '{"events":[],"events":[]}',
    '{"events":NaN}',
    '{"events":[],"explanation":"done"}',
    '{"events":[{"event_type":"企业收购","arguments":[]}]}',
    '{"events":[{"event_type":"企业收购","arguments":{"收购方":[]}}]}',
    '{"events":[{"event_type":"企业收购","arguments":{"收购方":[],"被收购方":[],"交易金额":[]}}]}',
])
def test_bad_output_is_not_repaired_into_empty_success(answer):
    result = inspect_answer(answer, "公告")
    assert result.events is None
    assert result.error is not None


def test_real_negative_is_a_valid_empty_event_list():
    result = inspect_answer('{"events":[]}', "公司发布会议通知。")
    assert result.events == []
    assert result.error is None


def test_unsupported_value_is_reported_without_deleting_model_prediction():
    answer = json.dumps(positive(), ensure_ascii=False)
    result = inspect_answer(answer, "甲公司发布公告。")
    assert result.events == positive()["events"]
    assert result.warnings == [{
        "code": "value_not_in_source", "event_type": "企业收购", "role": "被收购方", "value": "乙公司",
    }]


def test_duplicate_values_are_invalid():
    value = positive()
    value["events"][0]["arguments"]["收购方"] = ["甲公司", "甲公司"]
    with pytest.raises(ValueError):
        parse_answer(json.dumps(value))
