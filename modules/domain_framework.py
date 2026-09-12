"""
SIH26170 - Universal Domain Framework
Version: 3.0

Purpose:
    Provides a reusable domain framework for the application layer.

Supported domains:
    - electronics
    - mechanical
    - automotive
    - manufacturing
    - energy
    - aerospace
    - medical_equipment
    - general_unknown

Design:
    This module is independent of the Phase-1 AI core.
    It provides domain metadata, aliases, parameter categories,
    and domain resolution utilities for Phase-2 application logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DomainFramework:
    """Universal domain framework for SIH26170."""

    VERSION = "3.0"

    DEFAULT_DOMAIN = "general_unknown"

    SUPPORTED_DOMAINS = [
        "electronics",
        "mechanical",
        "automotive",
        "manufacturing",
        "energy",
        "aerospace",
        "medical_equipment",
        "general_unknown",
    ]

    # -------------------------------------------------------------
    # Domain metadata
    # -------------------------------------------------------------

    DOMAIN_METADATA: Dict[str, Dict[str, Any]] = {

        "electronics": {
            "display_name": "Electronics",
            "description": (
                "Electronic component and semiconductor "
                "measurement screening."
            ),
            "parameter_categories": [
                "electrical",
                "timing",
                "reliability",
            ],
            "known_parameters": [
                "Iddq",
                "Leakage",
                "Delay",
            ],
        },

        "mechanical": {
            "display_name": "Mechanical",
            "description": (
                "Mechanical component measurement and "
                "condition screening."
            ),
            "parameter_categories": [
                "dimensional",
                "mechanical",
                "thermal",
            ],
            "known_parameters": [],
        },

        "automotive": {
            "display_name": "Automotive",
            "description": (
                "Automotive component and system "
                "measurement screening."
            ),
            "parameter_categories": [
                "electrical",
                "mechanical",
                "thermal",
                "reliability",
            ],
            "known_parameters": [],
        },

        "manufacturing": {
            "display_name": "Manufacturing",
            "description": (
                "Manufacturing process and quality "
                "measurement screening."
            ),
            "parameter_categories": [
                "process",
                "quality",
                "dimensional",
                "thermal",
            ],
            "known_parameters": [],
        },

        "energy": {
            "display_name": "Energy",
            "description": (
                "Energy equipment and system "
                "measurement screening."
            ),
            "parameter_categories": [
                "electrical",
                "thermal",
                "power",
                "efficiency",
            ],
            "known_parameters": [],
        },

        "aerospace": {
            "display_name": "Aerospace",
            "description": (
                "Aerospace component and system "
                "measurement screening."
            ),
            "parameter_categories": [
                "electrical",
                "mechanical",
                "thermal",
                "reliability",
            ],
            "known_parameters": [],
        },

        "medical_equipment": {
            "display_name": "Medical Equipment",
            "description": (
                "Medical equipment component and "
                "system measurement screening."
            ),
            "parameter_categories": [
                "electrical",
                "mechanical",
                "thermal",
                "safety",
            ],
            "known_parameters": [],
        },

        "general_unknown": {
            "display_name": "General / Unknown",
            "description": (
                "Generic domain used when a reliable "
                "engineering domain cannot be determined."
            ),
            "parameter_categories": [
                "generic",
            ],
            "known_parameters": [],
        },
    }

    # -------------------------------------------------------------
    # Domain aliases
    # -------------------------------------------------------------

    DOMAIN_ALIASES: Dict[str, str] = {

        "electronics": "electronics",
        "electronic": "electronics",
        "semiconductor": "electronics",
        "semiconductors": "electronics",
        "vlsi": "electronics",
        "asic": "electronics",
        "fpga": "electronics",
        "chip": "electronics",
        "chips": "electronics",
        "ic": "electronics",
        "ics": "electronics",

        "mechanical": "mechanical",
        "mechanics": "mechanical",
        "machine": "mechanical",
        "machinery": "mechanical",

        "automotive": "automotive",
        "automobile": "automotive",
        "vehicle": "automotive",
        "vehicles": "automotive",
        "car": "automotive",

        "manufacturing": "manufacturing",
        "manufacture": "manufacturing",
        "production": "manufacturing",
        "factory": "manufacturing",
        "industrial": "manufacturing",

        "energy": "energy",
        "power": "energy",
        "battery": "energy",
        "batteries": "energy",
        "solar": "energy",

        "aerospace": "aerospace",
        "aviation": "aerospace",
        "aircraft": "aerospace",
        "space": "aerospace",

        "medical": "medical_equipment",
        "medical_equipment": "medical_equipment",
        "healthcare": "medical_equipment",
        "hospital": "medical_equipment",
    }

    # -------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------

    def __init__(
        self,
        supported_domains: Optional[List[str]] = None,
    ):

        if supported_domains is None:

            self.supported_domains = list(
                self.SUPPORTED_DOMAINS
            )

        else:

            normalized = []

            for domain in supported_domains:

                canonical = self.normalize_domain(
                    domain
                )

                if canonical not in normalized:
                    normalized.append(canonical)

            if self.DEFAULT_DOMAIN not in normalized:
                normalized.append(
                    self.DEFAULT_DOMAIN
                )

            self.supported_domains = normalized

    # -------------------------------------------------------------
    # Normalize domain
    # -------------------------------------------------------------

    @classmethod
    def normalize_domain(
        cls,
        domain: Any
    ) -> str:
        """
        Convert a domain name or alias into canonical form.
        """

        if domain is None:
            return cls.DEFAULT_DOMAIN

        value = str(domain).strip().lower()

        if not value:
            return cls.DEFAULT_DOMAIN

        value = value.replace(
            "-",
            "_"
        )

        value = value.replace(
            " ",
            "_"
        )

        if value in cls.SUPPORTED_DOMAINS:
            return value

        return cls.DOMAIN_ALIASES.get(
            value,
            cls.DEFAULT_DOMAIN
        )

    # -------------------------------------------------------------
    # Check domain
    # -------------------------------------------------------------

    def is_supported(
        self,
        domain: Any
    ) -> bool:
        """Return True if the domain is supported."""

        canonical = self.normalize_domain(
            domain
        )

        return canonical in self.supported_domains

    # -------------------------------------------------------------
    # Resolve domain
    # -------------------------------------------------------------

    def resolve(
        self,
        domain: Any = None
    ) -> str:
        """Return the canonical supported domain."""

        canonical = self.normalize_domain(
            domain
        )

        if canonical in self.supported_domains:
            return canonical

        return self.DEFAULT_DOMAIN

    # -------------------------------------------------------------
    # Get metadata
    # -------------------------------------------------------------

    def get_metadata(
        self,
        domain: Any
    ) -> Dict[str, Any]:
        """Return metadata for a domain."""

        canonical = self.resolve(
            domain
        )

        metadata = self.DOMAIN_METADATA.get(
            canonical,
            self.DOMAIN_METADATA[
                self.DEFAULT_DOMAIN
            ]
        )

        return {
            "domain": canonical,
            **metadata,
        }

    # -------------------------------------------------------------
    # Display name
    # -------------------------------------------------------------

    def get_display_name(
        self,
        domain: Any
    ) -> str:
        """Return human-readable domain name."""

        metadata = self.get_metadata(
            domain
        )

        return str(
            metadata.get(
                "display_name",
                "General / Unknown"
            )
        )

    # -------------------------------------------------------------
    # Description
    # -------------------------------------------------------------

    def get_description(
        self,
        domain: Any
    ) -> str:
        """Return domain description."""

        metadata = self.get_metadata(
            domain
        )

        return str(
            metadata.get(
                "description",
                ""
            )
        )

    # -------------------------------------------------------------
    # Parameter categories
    # -------------------------------------------------------------

    def get_parameter_categories(
        self,
        domain: Any
    ) -> List[str]:
        """Return parameter categories for a domain."""

        metadata = self.get_metadata(
            domain
        )

        categories = metadata.get(
            "parameter_categories",
            []
        )

        return list(categories)

    # -------------------------------------------------------------
    # Known parameters
    # -------------------------------------------------------------

    def get_known_parameters(
        self,
        domain: Any
    ) -> List[str]:
        """Return registered domain-specific parameters."""

        metadata = self.get_metadata(
            domain
        )

        parameters = metadata.get(
            "known_parameters",
            []
        )

        return list(parameters)

    # -------------------------------------------------------------
    # Domain matching
    # -------------------------------------------------------------

    def match_domain(
        self,
        text: Any
    ) -> str:
        """
        Resolve a domain from free-text input.

        This is a lightweight deterministic matcher.
        Automatic scoring can be added by the Phase-2
        domain detection layer.
        """

        if text is None:
            return self.DEFAULT_DOMAIN

        value = str(text).strip().lower()

        if not value:
            return self.DEFAULT_DOMAIN

        normalized = value.replace(
            "-",
            " "
        ).replace(
            "_",
            " "
        )

        # Exact / alias matching first.
        direct = self.DOMAIN_ALIASES.get(
            value
        )

        if direct:
            return direct

        # Keyword matching.
        keyword_groups = {

            "electronics": [
                "electronics",
                "electronic",
                "semiconductor",
                "semiconductors",
                "vlsi",
                "asic",
                "fpga",
                "chip",
                "integrated circuit",
                "iddq",
                "leakage",
                "propagation delay",
            ],

            "mechanical": [
                "mechanical",
                "mechanics",
                "machine",
                "machinery",
                "dimension",
                "bearing",
                "shaft",
                "vibration",
            ],

            "automotive": [
                "automotive",
                "automobile",
                "vehicle",
                "car",
                "engine",
                "ev",
            ],

            "manufacturing": [
                "manufacturing",
                "manufacture",
                "production",
                "factory",
                "process",
                "assembly",
                "quality control",
            ],

            "energy": [
                "energy",
                "power",
                "battery",
                "solar",
                "voltage",
                "current",
                "energy storage",
            ],

            "aerospace": [
                "aerospace",
                "aviation",
                "aircraft",
                "spacecraft",
                "space",
                "satellite",
            ],

            "medical_equipment": [
                "medical",
                "healthcare",
                "hospital",
                "medical equipment",
                "diagnostic equipment",
                "patient monitor",
            ],
        }

        scores = {
            domain: 0
            for domain in keyword_groups
        }

        for domain, keywords in keyword_groups.items():

            for keyword in keywords:

                if keyword in normalized:
                    scores[domain] += 1

        if not scores:
            return self.DEFAULT_DOMAIN

        best_domain = max(
            scores,
            key=scores.get
        )

        if scores[best_domain] <= 0:
            return self.DEFAULT_DOMAIN

        return self.normalize_domain(
            best_domain
        )

    # -------------------------------------------------------------
    # Framework information
    # -------------------------------------------------------------

    def get_framework_info(
        self
    ) -> Dict[str, Any]:
        """Return complete framework metadata."""

        return {
            "version": self.VERSION,
            "default_domain":
                self.DEFAULT_DOMAIN,
            "supported_domains":
                list(self.supported_domains),
            "domain_count":
                len(self.supported_domains),
        }

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    def summary(
        self
    ) -> Dict[str, Any]:
        """Return concise framework summary."""

        return {
            "version": self.VERSION,
            "default_domain":
                self.DEFAULT_DOMAIN,
            "supported_domains":
                list(self.supported_domains),
            "domain_count":
                len(self.supported_domains),
        }