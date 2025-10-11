import shlex
from dataclasses import dataclass, fields
from types import UnionType
from typing import Any, Type, Union, get_args, get_origin


@dataclass
class SubargsBase:
    """To use subargs, extend this class and define the fields that you need"""

    @classmethod
    def fields(cls):
        """Return a map of field name to type name, e.g. 'id': 'int'"""

        def resolve_type(tp):
            # Older style unions like Union[FooType, BarType] result in typing.Union type. Newer
            # style unions written as FooType | BarType result in types.UnionType. Eventually
            # we should switch to all new style unions, but for now we support both.
            if get_origin(tp) in (Union, UnionType):
                args = [arg for arg in get_args(tp) if arg is not type(None)]
                return args[0].__name__ if args else None
            return tp.__name__

        return {f.name: resolve_type(f.type) for f in fields(cls)}


class ParseSubargsError(ValueError):
    pass


def split_with_shlex(s: str, separator: str):
    lexer = shlex.shlex(s, posix=True)
    lexer.whitespace = separator
    lexer.whitespace_split = True
    return list(lexer)


def parse_subargs(s: str, cls: Type[SubargsBase], separator=",", assign="=", ignore_fields=set()):
    """Parses the string s and uses it to instantiate a class cls and return it."""
    if s == "":
        return cls()
    class_fields = cls.fields()
    args = {}

    for part in split_with_shlex(s, separator):
        if assign not in part:
            raise ParseSubargsError(f"The subargument '{part}' needs to have an assignment using '{assign}'")

        k, v = part.split(assign)
        if k in ignore_fields:
            continue

        if k not in class_fields:
            raise ParseSubargsError(
                f"Field '{k}' not in class {cls.__name__}.  Available fields are: {', '.join(class_fields.keys())}"
            )

        field_type = class_fields[k]
        if field_type == "str":
            pass
        elif field_type == "int":
            v = int(v)
        elif field_type == "bool":
            v = v.lower() in ("true", "1", "yes")
        elif field_type == "float":
            v = float(v)
        else:
            raise ParseSubargsError(
                f"Field '{k}' is of type '{field_type}' that I don't know how to parse. May need to update me."
            )
        args[k] = v

    return cls(**args)


def override_subargs(s: str, override: dict[str, Any], separator=",", assign="=") -> str:
    """
    Overrides or appends key-value pairs in a subargument string.

    Parses a string of subarguments (e.g., "a=1,b=2") and replaces the values of keys found in the `override` dictionary.
    Keys in `override` that are not present in the input string are appended as new key-value pairs.
    The separator and assignment characters can be customized.

    Args:
        s (str): The input subargument string containing key-value pairs.
        override (dict[str, Any]): Dictionary of keys and values to override or append.
        separator (str, optional): Character used to separate key-value pairs. Defaults to ",".
        assign (str, optional): Character used to assign values to keys. Defaults to "=".

    Returns:
        str: The resulting subargument string with overrides and additions applied.

    Raises:
        ParseSubargsError: If any part of the input string does not contain the assignment character.
    """
    fields = []
    unused_keys = set(override.keys())
    for part in split_with_shlex(s, separator):
        if assign not in part:
            raise ParseSubargsError(f"The subargument '{part}' needs to have an assignment using '{assign}'")

        k, v = part.split(assign)
        if k in override:
            v = str(override[k])
            unused_keys.remove(k)

        fields.append(f"{k}{assign}{v}")

    for key in unused_keys:
        fields.append(f"{key}{assign}{override[key]}")

    return separator.join(fields)
