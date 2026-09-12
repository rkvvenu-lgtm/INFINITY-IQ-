from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Central input-data loader for SIH26170.

    Supported formats:
        - CSV
        - XLSX
        - XLS

    The loader can also normalize an already-created
    pandas DataFrame for programmatic pipeline usage.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv",
        ".xlsx",
        ".xls"
    }

    # ============================================================
    # FILE LOADING
    # ============================================================

    def load(self, source):
        """
        Load a dataset from a file path or pandas DataFrame.

        Parameters
        ----------
        source : str, Path, or pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """

        if isinstance(source, pd.DataFrame):
            return self._prepare_dataframe(source)

        if source is None:
            raise ValueError(
                "No dataset source was provided."
            )

        path = Path(source).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Dataset path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(
                sorted(self.SUPPORTED_EXTENSIONS)
            )

            raise ValueError(
                f"Unsupported dataset format '{extension}'. "
                f"Supported formats: {supported}"
            )

        try:
            if extension == ".csv":
                data = pd.read_csv(path)

            elif extension in {".xlsx", ".xls"}:
                data = pd.read_excel(path)

            else:
                raise ValueError(
                    f"Unsupported dataset format: {extension}"
                )

        except Exception as exc:
            raise ValueError(
                f"Unable to read dataset '{path}': {exc}"
            ) from exc

        return self._prepare_dataframe(data)

    # ============================================================
    # DATAFRAME PREPARATION
    # ============================================================

    @staticmethod
    def _prepare_dataframe(data):
        """
        Perform basic structural normalization.

        No engineering decision or ML transformation is
        performed here.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Dataset must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Dataset contains no rows."
            )

        result = data.copy()

        # Normalize column names while preserving meaning.
        result.columns = [
            str(column).strip()
            for column in result.columns
        ]

        # Remove completely empty columns.
        result = result.dropna(
            axis=1,
            how="all"
        )

        if result.empty:
            raise ValueError(
                "Dataset contains no usable columns."
            )

        return result

    # ============================================================
    # FILE INFORMATION
    # ============================================================

    def get_file_metadata(self, source):
        """
        Return basic metadata about a dataset source.

        This metadata can later be used by:
            - Streamlit UI
            - reporting
            - audit records
            - dataset identification
        """

        if isinstance(source, pd.DataFrame):

            data = self._prepare_dataframe(
                source
            )

            return {
                "source_type": "DATAFRAME",
                "file_name": None,
                "file_extension": None,
                "rows": int(len(data)),
                "columns": int(len(data.columns)),
                "column_names": list(data.columns)
            }

        path = Path(source).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        data = self.load(path)

        return {
            "source_type": "FILE",
            "file_name": path.name,
            "file_extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
            "column_names": list(data.columns)
        }

    # ============================================================
    # DATASET ID
    # ============================================================

    @staticmethod
    def generate_dataset_id(data):
        """
        Generate a deterministic identifier from dataset content.

        The identifier is useful for reporting and audit trails.
        """

        import hashlib

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Dataset must be a pandas DataFrame."
            )

        normalized = data.copy()

        normalized.columns = [
            str(column)
            for column in normalized.columns
        ]

        content = normalized.to_csv(
            index=False
        ).encode("utf-8")

        return hashlib.sha256(
            content
        ).hexdigest()[:16]

    # ============================================================
    # VALIDATION HELPERS
    # ============================================================

    @classmethod
    def is_supported_file(cls, source):
        """
        Return True when the supplied file has a supported
        dataset extension.
        """

        if source is None:
            return False

        if isinstance(source, pd.DataFrame):
            return True

        try:
            extension = Path(
                source
            ).suffix.lower()

        except Exception:
            return False

        return extension in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def supported_extensions(cls):
        """
        Return supported file extensions.
        """

        return sorted(
            cls.SUPPORTED_EXTENSIONS
        )