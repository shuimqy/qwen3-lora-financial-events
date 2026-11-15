"""The experiment's strict protocol: never repair or silently discard output."""
import json
from dataclasses import dataclass

SCHEMA = {
    "企业收购": ["收购方", "被收购方", "交易金额"],
    "股份回购": ["回购方", "回购股份数量", "交易金额"],
}


class InvalidOutput(ValueError):
    pass


def unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidOutput("Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value: str):
    raise InvalidOutput("Nonfinite JSON constant: " + value)


def parse_answer(answer: str) -> list[dict]:
    result = json.loads(answer, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(result, dict) or set(result) != {"events"}:
        raise InvalidOutput("Expected an object with only the events key")
    events = result["events"]
    if not isinstance(events, list) or len(events) > 1:
        raise InvalidOutput("Expected an events array with at most one event")
    for event in events:
        if not isinstance(event, dict) or set(event) != {"event_type", "arguments"}:
            raise InvalidOutput("Invalid event keys")
        kind = event["event_type"]
        if not isinstance(kind, str) or kind not in SCHEMA:
            raise InvalidOutput("Unknown event type")
        fields = event["arguments"]
        if not isinstance(fields, dict) or set(fields) != set(SCHEMA[kind]):
            raise InvalidOutput("Missing or unknown argument role")
        for values in fields.values():
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise InvalidOutput("Argument values must be arrays of nonempty strings")
            if len(set(values)) != len(values):
                raise InvalidOutput("Duplicate argument value")
        if not any(fields.values()):
            raise InvalidOutput("Positive event has no arguments")
    return events


@dataclass
class ParsedOutput:
    events: list[dict] | None
    error: str | None
    warnings: list[dict]


def inspect_answer(answer: str, article: str) -> ParsedOutput:
    try:
        events = parse_answer(answer)
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        return ParsedOutput(events=None, error=str(error), warnings=[])
    warnings = []
    for event in events:
        for role, values in event["arguments"].items():
            for value in values:
                if value not in article:
                    warnings.append({
                        "code": "value_not_in_source",
                        "event_type": event["event_type"],
                        "role": role,
                        "value": value,
                    })
    return ParsedOutput(events=events, error=None, warnings=warnings)
