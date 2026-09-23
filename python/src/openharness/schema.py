"""JSON Schema helpers: build schemas from Python types, validate values, and
parse model output into typed objects.

Supported types: str, int, float, bool, None, list[T], dict[str, T], tuple,
Optional/Union, Literal, Enum, dataclasses, TypedDict, and Pydantic models
(when Pydantic is installed). No third-party dependency is required.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import json
import re
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

_PRIMITIVES: dict[Any, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    type(None): {"type": "null"},
    Any: {},
}


def _is_pydantic(tp: Any) -> bool:
    return inspect.isclass(tp) and hasattr(tp, "model_json_schema") and hasattr(tp, "model_validate")


def _is_typeddict(tp: Any) -> bool:
    return inspect.isclass(tp) and issubclass(tp, dict) and hasattr(tp, "__annotations__") and hasattr(tp, "__total__")


def type_to_schema(tp: Any, *, description: str | None = None) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    schema = _type_to_schema(tp)
    if description:
        schema = {**schema, "description": description}
    return schema


def _type_to_schema(tp: Any) -> dict[str, Any]:
    if tp is inspect.Parameter.empty:
        return {}
    if isinstance(tp, dict):  # already a schema
        return tp
    if tp in _PRIMITIVES:
        return dict(_PRIMITIVES[tp])
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is typing.Annotated:
        base = _type_to_schema(args[0])
        for meta in args[1:]:
            if isinstance(meta, str):
                base["description"] = meta
        return base
    if origin is Literal:
        values = list(args)
        s: dict[str, Any] = {"enum": values}
        kinds = {type(v) for v in values}
        if len(kinds) == 1 and next(iter(kinds)) in _PRIMITIVES:
            s["type"] = _PRIMITIVES[next(iter(kinds))]["type"]
        return s
    if origin in (Union, types.UnionType):
        non_null = [a for a in args if a is not type(None)]
        subs = [_type_to_schema(a) for a in non_null]
        if len(non_null) < len(args):
            subs.append({"type": "null"})
        if all(set(s) == {"type"} and isinstance(s["type"], str) for s in subs):
            return {"type": [s["type"] for s in subs]}
        return {"anyOf": subs}
    if origin in (list, set, frozenset, typing.Sequence, typing.Iterable) or tp in (list, set):
        return {"type": "array", "items": _type_to_schema(args[0]) if args else {}}
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _type_to_schema(args[0])}
        return {
            "type": "array",
            "prefixItems": [_type_to_schema(a) for a in args],
            "minItems": len(args),
            "maxItems": len(args),
        }
    if origin in (dict, typing.Mapping) or tp is dict:
        s = {"type": "object"}
        if len(args) == 2:
            s["additionalProperties"] = _type_to_schema(args[1])
        return s
    if inspect.isclass(tp) and issubclass(tp, enum.Enum):
        return {"enum": [m.value for m in tp]}
    if _is_pydantic(tp):
        return _inline_refs(tp.model_json_schema())
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return _object_schema(tp, {f.name: f for f in dataclasses.fields(tp)})
    if _is_typeddict(tp):
        return _object_schema(tp, None)
    raise TypeError(f"Cannot build a JSON schema for type {tp!r}")


def _object_schema(tp: Any, dc_fields: dict[str, dataclasses.Field] | None) -> dict[str, Any]:
    hints = get_type_hints(tp, include_extras=True)
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, hint in hints.items():
        props[name] = _type_to_schema(hint)
        if dc_fields is not None:
            f = dc_fields.get(name)
            if f is not None and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                required.append(name)
            elif f is not None and f.default is not dataclasses.MISSING:
                props[name].setdefault("default", f.default)
        else:  # TypedDict
            if name in getattr(tp, "__required_keys__", hints.keys()):
                required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = required
    doc = inspect.getdoc(tp)
    if doc and not doc.startswith(tp.__name__ + "("):
        schema["description"] = doc
    return schema


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs`` references (some providers reject ``$ref``)."""
    defs = schema.get("$defs") or schema.get("definitions") or {}

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 30:
            return node
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
                target = defs.get(ref.split("/")[-1], {})
                merged = {**resolve(target, depth + 1), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: resolve(v, depth + 1) for k, v in node.items() if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        return node

    return resolve(schema)


# ---------------------------------------------------------------- strict mode


def is_strict_compatible(schema: dict[str, Any]) -> bool:
    """True when every object lists all properties as required and forbids extras.

    OpenAI's strict mode requires this. Optional fields must be expressed as a
    union with null instead.
    """
    if not isinstance(schema, dict):
        return True
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties", {})
        if schema.get("additionalProperties", True) is not False:
            return False
        if set(schema.get("required", [])) != set(props):
            return False
    for key in ("properties", "$defs"):
        for sub in (schema.get(key) or {}).values():
            if not is_strict_compatible(sub):
                return False
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict) and not is_strict_compatible(schema[key]):
            return False
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for sub in schema.get(key, []) or []:
            if not is_strict_compatible(sub):
                return False
    return True


# ---------------------------------------------------------------- validation


class SchemaValidationError(ValueError):
    pass


_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate ``value`` against a practical subset of JSON Schema.

    Covers type, enum, const, properties, required, additionalProperties,
    items, prefixItems, anyOf/oneOf/allOf, min/max length and items,
    minimum/maximum and pattern. Raises ``SchemaValidationError``.
    """
    if not schema:
        return
    if "anyOf" in schema or "oneOf" in schema:
        options = schema.get("anyOf") or schema.get("oneOf")
        errors = []
        for opt in options:
            try:
                validate(value, opt, path)
                break
            except SchemaValidationError as e:
                errors.append(str(e))
        else:
            raise SchemaValidationError(f"{path}: does not match any allowed shape ({'; '.join(errors[:3])})")
    for sub in schema.get("allOf", []):
        validate(value, sub, path)
    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: {value!r} is not one of {schema['enum']!r}")
    t = schema.get("type")
    if t is not None:
        allowed = t if isinstance(t, list) else [t]
        if not any(_is_type(value, a) for a in allowed):
            raise SchemaValidationError(f"{path}: expected {'/'.join(allowed)}, got {type(value).__name__}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                raise SchemaValidationError(f"{path}: missing required property '{req}'")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                validate(v, props[k], f"{path}.{k}")
            elif extra is False:
                raise SchemaValidationError(f"{path}: unexpected property '{k}'")
            elif isinstance(extra, dict):
                validate(v, extra, f"{path}.{k}")
    if isinstance(value, list):
        prefix = schema.get("prefixItems")
        if prefix:
            for i, (v, s) in enumerate(zip(value, prefix, strict=False)):
                validate(v, s, f"{path}[{i}]")
        elif isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                validate(v, schema["items"], f"{path}[{i}]")
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaValidationError(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaValidationError(f"{path}: allows at most {schema['maxItems']} items")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaValidationError(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaValidationError(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise SchemaValidationError(f"{path}: does not match pattern {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path}: must be <= {schema['maximum']}")


def _is_type(value: Any, t: str) -> bool:
    if t == "integer":
        return (
            isinstance(value, int) and not isinstance(value, bool) or (isinstance(value, float) and value.is_integer())
        )
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    py = _JSON_TYPES.get(t)
    return py is None or isinstance(value, py)


# ---------------------------------------------------------------- output types


def output_schema_for(output_type: Any) -> dict[str, Any]:
    """Schema for an agent ``output_type``. Non-object types are wrapped in {"value": ...}."""
    schema = type_to_schema(output_type)
    if schema.get("type") != "object":
        return {"type": "object", "properties": {"value": schema}, "required": ["value"], "additionalProperties": False}
    return schema


def parse_output(text: str, output_type: Any, schema: dict[str, Any]) -> Any:
    """Parse and validate the model's final text into ``output_type``."""
    data = extract_json(text)
    validate(data, schema)
    wrapped = type_to_schema(output_type).get("type") != "object"
    if wrapped:
        data = data["value"]
    return coerce(data, output_type)


def extract_json(text: str) -> Any:
    """Parse JSON from model text, tolerating code fences and surrounding prose."""
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = s.find(open_c), s.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise SchemaValidationError("Response was not valid JSON")


def coerce(data: Any, tp: Any) -> Any:
    """Turn validated JSON data into an instance of ``tp``."""
    if isinstance(tp, dict) or tp is Any or tp is inspect.Parameter.empty:
        return data
    if _is_pydantic(tp):
        return tp.model_validate(data)
    if dataclasses.is_dataclass(tp) and isinstance(tp, type) and isinstance(data, dict):
        hints = get_type_hints(tp)
        return tp(**{k: coerce(v, hints.get(k, Any)) for k, v in data.items() if k in hints})
    if inspect.isclass(tp) and issubclass(tp, enum.Enum):
        return tp(data)
    origin, args = get_origin(tp), get_args(tp)
    if origin is typing.Annotated:
        return coerce(data, args[0])
    if origin in (list, set, frozenset) and isinstance(data, list):
        items = [coerce(v, args[0]) if args else v for v in data]
        return origin(items) if origin is not list else items
    if origin is tuple and isinstance(data, list):
        return tuple(data)
    if origin in (dict, typing.Mapping) and isinstance(data, dict) and len(args) == 2:
        return {k: coerce(v, args[1]) for k, v in data.items()}
    if origin in (Union, types.UnionType):
        if data is None:
            return None
        for a in args:
            if a is type(None):
                continue
            try:
                validate(data, type_to_schema(a))
                return coerce(data, a)
            except (SchemaValidationError, TypeError):
                continue
        return data
    if tp is float and isinstance(data, int):
        return float(data)
    if tp is int and isinstance(data, float) and data.is_integer():
        return int(data)
    return data


def schema_name(output_type: Any) -> str:
    name = getattr(output_type, "__name__", None) or "output"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]
