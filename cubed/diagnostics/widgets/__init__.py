try:
    from cubed.diagnostics.widgets.core import (
        FILTERS,
        TEMPLATE_PATHS,
        get_environment,
        get_template,
    )

    from .memory import LiveMemoryViewer, MemoryWidget
    from .plan import LivePlanViewer, PlanWidget

    __all__ = ["LiveMemoryViewer", "LivePlanViewer", "MemoryWidget", "PlanWidget"]

except ImportError as e:
    msg = (
        "Cubed diagnostics requirements are not installed.\n\n"
        "Please either conda or pip install as follows:\n\n"
        "  conda install cubed                     # either conda install\n"
        '  python -m pip install "cubed[diagnostics]" --upgrade  # or python -m pip install'
    )
    exception = e  # Explicit reference for e as it will be lost outside the try block
    FILTERS = {}
    TEMPLATE_PATHS = []

    def get_environment():  # type: ignore
        raise ImportError(msg) from exception

    def get_template(name: str):  # type: ignore
        raise ImportError(msg) from exception
