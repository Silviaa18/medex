import pathlib

import joblib
import pandas as pd
from abc import ABC, abstractmethod
from medex.database_schema import TableNumerical, TableCategorical, NameType
from medex.dto.entity import EntityType
from typing import Union, Optional
from sqlalchemy.orm import aliased, Query
from sqlalchemy import and_
import numpy as np


class PluginInterface:
    def on_loaded(self):
        pass

    def on_db_ready(self, session):
        pass

    def on_stopped(self):
        pass

    # ... Expand with further lifecycle methods


