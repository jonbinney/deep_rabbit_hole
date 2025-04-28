from dataclasses import dataclass, fields
from typing import Type, Union, get_args, get_origin


@dataclass
class SubargsBase:
    """To use subargs, extend this class and define the fields that you need"""

    @classmethod
    def fields(cls):
        """Return a map of field name to type name, e.g. 'id': 'int'"""

        def resolve_type(tp):
            if get_origin(tp) is Union:
                args = [arg for arg in get_args(tp) if arg is not type(None)]
                return args[0].__name__ if args else None
            return tp.__name__

        return {f.name: resolve_type(f.type) for f in fields(cls)}


class ParseSubargsError(ValueError):
    pass


def parse_subargs(s: str, cls: Type[SubargsBase], separator=",", assign="=", ignore_fields=set()):
    """Parses the string s and uses it to instantiate a class cls and return it."""
    if s == "":
        return cls()

    class_fields = cls.fields()
    args = {}
    for part in s.split(separator):
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
