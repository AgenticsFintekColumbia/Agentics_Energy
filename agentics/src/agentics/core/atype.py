import csv
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import pandas as pd
from pydantic import BaseModel, Field, create_model

from agentics.core.utils import sanitize_field_name


class AGString(BaseModel):
    string: Optional[str] = None


#####################################
####### Utils #######################


def pretty_print_atype(atype, indent: int = 2):
    """
    Recursively pretty print an 'atype' (Agentics/Pydantic typing model).
    Works on generics like list[int], dict[str, float], Optional[...], etc.
    """
    prefix = " " * indent

    origin = get_origin(atype)
    args = get_args(atype)

    if origin is None:
        # Base case: a plain class/type
        print(f"{prefix}{atype}")
    else:
        print(f"{prefix}{origin.__name__}[")
        for arg in args:
            pretty_print_atype(arg, indent + 2)
        print(f"{prefix}]")


def copy_attribute_values(
    state: BaseModel, source_attribute: str, target_attribute: str
) -> BaseModel:
    """for each state, copy the value from source_attribute to the target_attribute
    Usage: for generating fewshots,
    copy values for the target_attribute from source_attribute that holds the ground_truth.
    """
    source_value = getattr(state, source_attribute)
    setattr(state, target_attribute, source_value)
    return state

def get_pydantic_fields(atype: Type[BaseModel]) -> pd.DataFrame:
    rows = []
    for field_name, field in atype.model_fields.items():
        rows.append({
            "Field": field_name,
            "Type": str(field.annotation),         # Type annotation
            "Description": field.description       # Description from Field(...)
        })

    # Create DataFrame
    return pd.DataFrame(rows)

def get_active_fields(state: BaseModel, allowed_fields: Set[str] = None) -> Set[str]:
    """
    Returns the set of fields in `state` that are None and optionally intersect with allowed_fields.
    """
    active_fields = {
        k for k, v in state.model_dump().items() if v is not None and v != ""
    }
    return active_fields & allowed_fields if allowed_fields else active_fields


def pydantic_model_from_csv(file_path: str) -> type[BaseModel]:
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        columns = [sanitize_field_name(x) for x in reader.fieldnames]
        model_name = "AType#" + ":".join(columns)
        if not columns:
            raise ValueError("CSV file appears to have no header.")
        fields = {col: (Optional[str], None) for col in columns}
        return create_model(model_name, **fields)


def infer_pydantic_type(dtype: Any, sample_values: pd.Series = None) -> Any:
    if pd.api.types.is_integer_dtype(dtype):
        return Optional[int]
    elif pd.api.types.is_float_dtype(dtype):
        return Optional[float]
    elif pd.api.types.is_bool_dtype(dtype):
        return Optional[bool]
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return Optional[str]  # Or datetime.datetime
    elif sample_values is not None:
        # Check if the column contains lists of strings
        for val in sample_values:
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                return Optional[List[str]]
            elif isinstance(val, dict):
                if all(isinstance(k, str) for k in val.keys()):
                    if all(
                        isinstance(v, (str, list))
                        and (isinstance(v, str) or all(isinstance(i, str) for i in v))
                        for v in val.values()
                    ):
                        return Optional[Dict[str, Union[str, List[str]]]]
            break  # Only check the first non-null value
    return Optional[str]


def pydantic_model_from_dict(dict) -> type[BaseModel]:
    model_name = "AType#" + ":".join(dict.keys())
    fields = {}

    for col in dict.keys():
        sample_value = dict[col]
        pydantic_type = infer_pydantic_type(
            type(sample_value), sample_values=[sample_value]
        )
        fields[col] = (pydantic_type, Field(default=None))
    new_fields = {}
    for field, value in fields.items():
        new_fields[sanitize_field_name(field)] = value

    return create_model(model_name, **new_fields)


def pydantic_model_from_jsonl(
    file_path: str, sample_size: int = 100
) -> type[BaseModel]:
    df = pd.read_json(file_path, lines=True, nrows=sample_size, encoding="utf-8")

    model_name = "AType#" + ":".join(df.columns)
    fields = {}

    for col in df.columns:
        sample_values = df[col].head(5)
        pydantic_type = infer_pydantic_type(df[col].dtype, sample_values=sample_values)
        fields[col] = (pydantic_type, Field(default=None))
    new_fields = {}
    for field, value in fields.items():
        new_fields[sanitize_field_name(field)] = value

    return create_model(model_name, **new_fields)


def pydantic_model_from_dataframe(
    dataframe: pd.DataFrame, sample_size: int = 100
) -> Type[BaseModel]:
    df_sample = dataframe.head(sample_size)

    model_name = "AType#" + ":".join(df_sample.columns)
    fields = {}
    for col in df_sample.columns:
        pydantic_type = infer_pydantic_type(df_sample[col].dtype)
        fields[col] = (pydantic_type, Field(default=None))

    return create_model(model_name, **fields)


def create_pydantic_model(
    fields: List[Tuple[str, str, str, bool]], name: str = None
) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model from a list of field definitions.

    Args:
        fields: A list of (field_name, type_name, description) tuples.
        name: Optional name of the model.

    Returns:
        A dynamically created Pydantic model class.
    """
    type_mapping = {
        "string": str,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "Optional[str]": str,
        "Optional[int]": int,
        # Extend with more types as needed
    }

    if not name:
        model_name = "AType#" + ":".join([x[0] for x in fields])
    else:
        model_name = name

    field_definitions = {}
    print(fields)
    for field_name, type_name, description, required in fields:
        # ptype = type_mapping.get(model_name, str)  # default to str if unknown

        ptype = type_mapping[type_name] if type_name in type_mapping else Any
        if required:
            field_definitions[field_name] = (ptype, ...)
        else:
            field_definitions[field_name] = (Optional[ptype], None)
    return create_model(model_name, **field_definitions)


def make_all_fields_optional(
    model_cls: type[BaseModel], rename_type: str = None
) -> type[BaseModel]:
    """
    Returns a new Pydantic model class where all fields are Optional and default to None.

    Args:
        model_cls: Original Pydantic model class.
        rename_type: Name of the new model class (default: <OriginalName>Optional)

    Returns:
        New Pydantic model class with all fields optional.
    """
    fields = {}
    for name, field in model_cls.model_fields.items():
        # Original type
        annotation = field.annotation
        origin = get_origin(annotation)

        # Make it Optional if not already
        if origin is not Optional and annotation is not Any:
            annotation = Optional[annotation]

        fields[name] = (
            annotation,
            Field(default=None, title=field.title, description=field.description),
        )

    new_name = rename_type or f"{model_cls.__name__}Optional"
    return create_model(new_name, **fields)


def pretty_print_atype(atype, indent: int = 2):
    """
    Recursively pretty print an 'atype' (Agentics/Pydantic typing model).
    Works on generics like list[int], dict[str, float], Optional[...], etc.
    """
    prefix = " " * indent

    origin = get_origin(atype)
    args = get_args(atype)

    if origin is None:
        # Base case: a plain class/type
        print(f"{prefix}{atype}")
    else:
        print(f"{prefix}{origin.__name__}[")
        for arg in args:
            pretty_print_atype(arg, indent + 2)
        print(f"{prefix}]")
