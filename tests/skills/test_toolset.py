# Tencent is pleased to support the open source community by making tRPC-Agent-Python available.
#
# Copyright (C) 2026 Tencent. All rights reserved.
#
# tRPC-Agent-Python is licensed under Apache-2.0.
"""Unit tests for trpc_agent_sdk.skills._toolset.

Covers:
- SkillToolSet initialization
- SkillToolSet.get_tools: default set omits dynamic tool-selection helpers
- SkillToolSetWithDynamicTools.get_tools: opt-in skill_list_tools / skill_select_tools
- repository property
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trpc_agent_sdk.context import reset_invocation_ctx
from trpc_agent_sdk.context import set_invocation_ctx
from trpc_agent_sdk.skills._dynamic_toolset import SkillToolSetWithDynamicTools
from trpc_agent_sdk.skills._toolset import SkillToolSet


def _make_ctx():
    ctx = MagicMock()
    ctx.agent_context = MagicMock()
    ctx.agent_context.with_metadata = MagicMock()
    return ctx


class TestSkillToolSetInit:
    def test_default_init(self, tmp_path):
        ts = SkillToolSet(paths=[str(tmp_path)])
        assert ts.name == "skill_toolset"
        assert ts.repository is not None

    def test_custom_repository(self):
        mock_repo = MagicMock()
        mock_repo.workspace_runtime = MagicMock()
        ts = SkillToolSet(repository=mock_repo)
        assert ts.repository is mock_repo


class TestSkillToolSetGetTools:
    async def test_get_tools_returns_tools(self, tmp_path):
        ts = SkillToolSet(paths=[str(tmp_path)])
        ctx = _make_ctx()
        tools = await ts.get_tools(ctx)
        assert len(tools) > 0

    async def test_get_tools_includes_run_and_exec(self, tmp_path):
        ts = SkillToolSet(paths=[str(tmp_path)])
        ctx = _make_ctx()
        tools = await ts.get_tools(ctx)
        tool_names = [t.name for t in tools]
        assert "skill_run" in tool_names
        assert "skill_exec" in tool_names

    async def test_get_tools_includes_function_tools(self, tmp_path):
        ts = SkillToolSet(paths=[str(tmp_path)])
        ctx = _make_ctx()
        tools = await ts.get_tools(ctx)
        tool_names = [t.name for t in tools]
        assert "skill_load" in tool_names
        assert "skill_list" in tool_names
        assert "skill_list_docs" in tool_names
        assert "skill_select_docs" in tool_names
        assert "skill_list_tools" not in tool_names
        assert "skill_select_tools" not in tool_names

    async def test_get_tools_sets_metadata(self, tmp_path):
        ts = SkillToolSet(paths=[str(tmp_path)])
        ctx = _make_ctx()
        await ts.get_tools(ctx)
        ctx.agent_context.with_metadata.assert_called()


@pytest.mark.parametrize("toolset_cls", [SkillToolSet, SkillToolSetWithDynamicTools])
class TestSkillToolSetFiltering:

    async def test_name_filter_on_first_and_cached_calls(self, tmp_path, toolset_cls):
        allowed = ["skill_load", "skill_list", "workspace_exec"]
        ts = toolset_cls(paths=[str(tmp_path)], tool_filter=allowed, is_include_all_tools=False)

        for _ in range(2):
            tools = await ts.get_tools(_make_ctx())
            assert {tool.name for tool in tools} == set(allowed)
            assert len(tools) == len(allowed)

    async def test_predicate_uses_current_context_without_filtering_cache(self, tmp_path, toolset_cls):
        calls = []

        def predicate(tool, ctx):
            calls.append((tool.name, ctx))
            return tool.name in ctx.allowed_tools

        ts = toolset_cls(paths=[str(tmp_path)], tool_filter=predicate, is_include_all_tools=False)
        for allowed in (set(), {"skill_run"}, {"skill_load"}):
            ctx = _make_ctx()
            ctx.allowed_tools = allowed
            calls.clear()
            tools = await ts.get_tools(ctx)
            assert {tool.name for tool in tools} == allowed
            assert calls
            assert all(call_ctx is ctx for _, call_ctx in calls)

    async def test_predicate_receives_implicit_context(self, tmp_path, toolset_cls):
        predicate = MagicMock(side_effect=lambda tool, ctx: tool.name == "skill_load")
        ts = toolset_cls(paths=[str(tmp_path)], tool_filter=predicate, is_include_all_tools=False)
        ctx = _make_ctx()
        token = set_invocation_ctx(ctx)
        try:
            for _ in range(2):
                predicate.reset_mock()
                tools = await ts.get_tools()
                assert [tool.name for tool in tools] == ["skill_load"]
                assert predicate.called
                assert all(call.args[1] is ctx for call in predicate.call_args_list)
        finally:
            reset_invocation_ctx(token)

    @pytest.mark.parametrize("tool_filter,is_include_all_tools", [
        (None, False),
        ([], False),
        (["skill_load"], True),
        (lambda tool, ctx: False, True),
    ])
    async def test_unfiltered_behavior_is_preserved(self, tmp_path, toolset_cls, tool_filter, is_include_all_tools):
        ts = toolset_cls(paths=[str(tmp_path)], tool_filter=tool_filter, is_include_all_tools=is_include_all_tools)
        first = [tool.name for tool in await ts.get_tools(_make_ctx())]
        second = [tool.name for tool in await ts.get_tools(_make_ctx())]
        assert {"skill_load", "skill_run", "skill_exec", "workspace_exec", "skill_list"} <= set(first)
        assert first == second
        assert len(first) == len(set(first))


class TestSkillToolSetWithDynamicTools:
    async def test_get_tools_includes_dynamic_selection_helpers(self, tmp_path):
        ts = SkillToolSetWithDynamicTools(paths=[str(tmp_path)])
        ctx = _make_ctx()
        tools = await ts.get_tools(ctx)
        tool_names = [t.name for t in tools]
        assert "skill_load" in tool_names
        assert "skill_list" in tool_names
        assert "skill_list_tools" in tool_names
        assert "skill_select_tools" in tool_names

    async def test_get_tools_does_not_duplicate_helpers_on_second_call(self, tmp_path):
        ts = SkillToolSetWithDynamicTools(paths=[str(tmp_path)])
        ctx = _make_ctx()
        first = [t.name for t in await ts.get_tools(ctx)]
        second = [t.name for t in await ts.get_tools(ctx)]
        assert first.count("skill_list_tools") == 1
        assert first.count("skill_select_tools") == 1
        assert second.count("skill_list_tools") == 1
        assert second.count("skill_select_tools") == 1
