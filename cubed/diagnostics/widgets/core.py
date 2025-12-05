import html
import os.path

from jinja2 import Environment, FileSystemLoader, Template
from jinja2.exceptions import TemplateNotFound

from cubed.utils import format_int, memory_repr

FILTERS = {
    "format_int": format_int,
    "html_escape": html.escape,
    "memory_repr": memory_repr,
}

TEMPLATE_PATHS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")]


def get_environment() -> Environment:
    loader = FileSystemLoader(TEMPLATE_PATHS)
    environment = Environment(loader=loader)
    environment.filters.update(FILTERS)
    return environment


def get_template(name: str) -> Template:
    try:
        return get_environment().get_template(name)
    except TemplateNotFound as e:
        raise TemplateNotFound(f"Unable to find {name} in {TEMPLATE_PATHS}") from e
