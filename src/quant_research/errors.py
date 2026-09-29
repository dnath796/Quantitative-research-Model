"""Expected, actionable errors exposed by the research API."""


class ResearchError(ValueError):
    """Invalid data, configuration, or research specification."""


class ModelFitError(ResearchError):
    """A model could not produce a reliable forecast."""
