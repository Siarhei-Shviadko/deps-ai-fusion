import re
import types
from enum import Enum

__all__ = ["SchemaHint"]


class SchemaHint(str, Enum):
    DATE = "date"
    AMOUNT = "amount"
    ID = "id"
    BOOLEAN = "boolean"


_SCHEMA_VALIDATORS = types.MappingProxyType(
    {
        SchemaHint.DATE: (
            re.compile(r"^\d{4}-\d{2}-\d{2}$"),
            re.compile(r"\d{1,4}[.\-/]\d{1,2}[.\-/]\d{2,4}"),
        ),
        SchemaHint.AMOUNT: (
            re.compile(r"^[\d\s.,]+$"),
            re.compile(r"\d"),
        ),
        SchemaHint.ID: (
            re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-/._\s]*$"),
            re.compile(r"[A-Za-z0-9]"),
        ),
    }
)
