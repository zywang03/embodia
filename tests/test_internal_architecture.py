"""Compatibility tests for internal module boundaries."""

from __future__ import annotations

import unittest

import inferaxis as infra


class InternalArchitectureTests(unittest.TestCase):
    """Ensure new internal modules preserve legacy public imports."""

    def test_core_schema_internal_modules_reexport_compatible_objects(self) -> None:
        from inferaxis.core import command_kinds
        from inferaxis.core import schema as schema_module
        from inferaxis.core import schema_compat
        from inferaxis.core import schema_models
        from inferaxis.core import schema_validation

        self.assertIs(schema_models.Frame, infra.Frame)
        self.assertIs(schema_models.Command, infra.Command)
        self.assertIs(schema_models.Action, infra.Action)
        self.assertIs(command_kinds.BuiltinCommandKind, infra.BuiltinCommandKind)
        self.assertIs(schema_module.Frame, infra.Frame)
        self.assertIs(schema_module.Command, infra.Command)
        self.assertIs(schema_module.Action, infra.Action)
        self.assertIs(
            schema_validation.validate_action,
            schema_module.validate_action,
        )
        self.assertIs(
            schema_compat.ensure_action_supported_by_robot,
            schema_module.ensure_action_supported_by_robot,
        )

    def test_core_schema_legacy_module_exports_remain_complete(self) -> None:
        from inferaxis.core import schema as schema_module

        expected_names = {
            "Action",
            "BuiltinCommandKind",
            "COMMAND_KIND_REGISTRY",
            "CUSTOM_COMMAND_KIND_PREFIX",
            "ComponentSpec",
            "Command",
            "CommandKindSpec",
            "Frame",
            "KNOWN_COMPONENT_TYPES",
            "PolicyOutputSpec",
            "PolicySpec",
            "RobotSpec",
            "ensure_action_matches_policy_spec",
            "ensure_action_supported_by_robot",
            "get_command_kind_spec",
            "is_custom_command_kind_name",
            "is_known_command_kind",
            "register_command_kind",
            "validate_action",
            "validate_command",
            "validate_component_spec",
            "validate_frame",
            "validate_policy_output_spec",
            "validate_policy_spec",
            "validate_robot_spec",
        }

        self.assertEqual(set(schema_module.__all__), expected_names)

    def test_profiling_render_internal_modules_reexport_helpers(self) -> None:
        from inferaxis.runtime.inference.profiling import render as render_module
        from inferaxis.runtime.inference.profiling import render_async_trace
        from inferaxis.runtime.inference.profiling import render_runtime_html
        from inferaxis.runtime.inference.profiling import render_runtime_svg

        self.assertIs(
            render_async_trace._step_plot_points,
            render_module._step_plot_points,
        )
        self.assertIs(
            render_async_trace._async_buffer_trace_svg,
            render_module._async_buffer_trace_svg,
        )
        self.assertIs(
            render_runtime_svg._runtime_profile_svg,
            render_module._runtime_profile_svg,
        )
        self.assertIs(
            render_runtime_html._runtime_profile_html,
            render_module._runtime_profile_html,
        )
