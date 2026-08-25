"""User-editable HEMS controller for the single-home example."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


class HEMSController:
    """Baseline pass-through controller.

    Replace the body of ``controls_for_step`` with your HEMS logic. Returning
    an empty dictionary lets OCHRE use its native equipment controls.
    """

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Return equipment controls to apply at ``timestamp``."""
        return {}
