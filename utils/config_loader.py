from pathlib import Path
import json


class ConfigLoader:
    """
    Central configuration loader for SIH26170.

    Responsibilities:
        - Resolve project-relative paths
        - Load JSON configuration files
        - Validate JSON structure
        - Provide reusable configuration access
    """

    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

    # ============================================================
    # PATH HANDLING
    # ============================================================

    def resolve_path(self, file_path):
        """
        Resolve a project-relative or absolute file path.
        """

        path = Path(file_path)

        if path.is_absolute():
            return path

        return self.project_root / path

    # ============================================================
    # JSON LOADING
    # ============================================================

    def load(self, file_path):
        """
        Load a JSON configuration file.

        Parameters
        ----------
        file_path : str or Path
            Project-relative or absolute JSON path.

        Returns
        -------
        dict
            Parsed configuration.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.

        ValueError
            If JSON content is invalid or root is not an object.
        """

        path = self.resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Configuration path is not a file: {path}"
            )

        try:
            with path.open(
                "r",
                encoding="utf-8"
            ) as file:
                config = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in configuration file "
                f"'{path}': {exc}"
            ) from exc

        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration root must be a JSON object: {path}"
            )

        return config

    # ============================================================
    # SAVE JSON
    # ============================================================

    def save(self, file_path, config):
        """
        Save a configuration dictionary as formatted JSON.

        This is provided for controlled future configuration
        management and reporting workflows.
        """

        if not isinstance(config, dict):
            raise TypeError(
                "Configuration must be a dictionary."
            )

        path = self.resolve_path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                config,
                file,
                indent=2,
                ensure_ascii=False
            )

    # ============================================================
    # VALUE ACCESS
    # ============================================================

    @staticmethod
    def get(config, key, default=None):
        """
        Get a top-level configuration value safely.
        """

        if not isinstance(config, dict):
            return default

        return config.get(
            key,
            default
        )

    @staticmethod
    def get_nested(config, *keys, default=None):
        """
        Safely access nested configuration values.

        Example:
            loader.get_nested(
                config,
                "drift_prediction",
                "validation",
                "test_size",
                default=0.20
            )
        """

        current = config

        for key in keys:

            if not isinstance(current, dict):
                return default

            if key not in current:
                return default

            current = current[key]

        return current

    # ============================================================
    # EXISTENCE CHECK
    # ============================================================

    def exists(self, file_path):
        """
        Check whether a configuration file exists.
        """

        return self.resolve_path(
            file_path
        ).is_file()

    # ============================================================
    # PROJECT ROOT
    # ============================================================

    def get_project_root(self):
        """
        Return the resolved project root directory.
        """

        return self.project_root