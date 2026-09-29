# Tencent is pleased to support the open source community by making tRPC-Agent-Python available.
#
# Copyright (C) 2026 Tencent. All rights reserved.
#
# tRPC-Agent-Python is licensed under Apache-2.0.

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from trpc_agent_sdk.code_executors import DEFAULT_EXEC_YIELD_MS
from trpc_agent_sdk.code_executors import DEFAULT_IO_YIELD_MS
from trpc_agent_sdk.code_executors import DEFAULT_POLL_LINES
from trpc_agent_sdk.code_executors import DEFAULT_SESSION_TTL_SEC
from trpc_agent_sdk.skills.tools._skill_exec import ExecInput
from trpc_agent_sdk.skills.tools._skill_exec import PollSessionTool
from trpc_agent_sdk.skills.tools._skill_exec import SkillExecTool
from trpc_agent_sdk.skills.tools._skill_exec import WriteStdinTool
from trpc_agent_sdk.skills.tools._skill_exec import _close_session
from trpc_agent_sdk.skills.tools._skill_exec import _collect_final_result
from trpc_agent_sdk.skills.tools._skill_exec import _detect_interaction
from trpc_agent_sdk.skills.tools._skill_exec import _ExecSession
from trpc_agent_sdk.skills.tools._skill_exec import _has_selection_items
from trpc_agent_sdk.skills.tools._skill_exec import _last_non_empty_line
from trpc_agent_sdk.skills.tools._skill_exec import _start_session
from trpc_agent_sdk.skills.tools._skill_exec import create_exec_tools
from trpc_agent_sdk.skills.tools._skill_run import SkillRunFile
from trpc_agent_sdk.skills.tools._skill_run import SkillRunInput


def _make_exec_tool() -> SkillExecTool:
    run_tool = MagicMock()
    run_tool._repository = MagicMock()
    run_tool._timeout = 300.0
    run_tool._resolve_cwd = MagicMock(return_value="skills/test")
    run_tool._build_command = MagicMock(return_value=("bash", ["-lc", "echo hello"]))
    run_tool._prepare_outputs = AsyncMock(return_value=([], None))
    run_tool._attach_artifacts_if_requested = AsyncMock()
    run_tool._merge_manifest_artifact_refs = MagicMock()
    return SkillExecTool(run_tool)


class TestHelpers:
    def test_last_non_empty_line(self):
        assert _last_non_empty_line("a\n\nb\n") == "b"

    def test_has_selection_items(self):
        assert _has_selection_items("1. a\n2. b") is True
        assert _has_selection_items("1. a") is False

    def test_detect_interaction_prompt(self):
        ret = _detect_interaction("running", "Enter your name:")
        assert ret is not None
        assert ret.needs_input is True

    def test_detect_interaction_selection(self):
        ret = _detect_interaction("running", "Choose:\n1. A\n2. B\nEnter the number:")
        assert ret is not None
        assert ret.kind == "selection"


class TestModelsAndConstants:
    def test_exec_input_defaults(self):
        inp = ExecInput(skill="s", command="echo hi")
        assert inp.yield_time_ms == 0
        assert inp.poll_lines == 0
        assert inp.tty is False

    def test_default_constants(self):
        assert DEFAULT_EXEC_YIELD_MS > 0
        assert DEFAULT_IO_YIELD_MS > 0
        assert DEFAULT_POLL_LINES > 0
        assert DEFAULT_SESSION_TTL_SEC > 0


class TestSessionStore:
    @pytest.mark.asyncio
    async def test_put_get_remove(self):
        tool = _make_exec_tool()
        sess = MagicMock()
        sess.exited_at = None
        sess.proc.state = AsyncMock(return_value=MagicMock(status="running", exit_code=None))
        await tool._put_session("s1", sess)
        got = await tool._get_session("s1")
        assert got is sess
        removed = await tool._remove_session("s1")
        assert removed is sess


class TestFactoryAndDeclarations:
    def test_create_exec_tools(self):
        run_tool = MagicMock()
        run_tool._repository = MagicMock()
        run_tool._timeout = 300.0
        tools = create_exec_tools(run_tool)
        assert len(tools) == 4
        assert isinstance(tools[0], SkillExecTool)
        assert isinstance(tools[1], WriteStdinTool)
        assert isinstance(tools[2], PollSessionTool)

    def test_declaration_names(self):
        exec_tool = _make_exec_tool()
        assert exec_tool._get_declaration().name == "skill_exec"
        assert WriteStdinTool(exec_tool)._get_declaration().name == "skill_write_stdin"
        assert PollSessionTool(exec_tool)._get_declaration().name == "skill_poll_session"


class TestCloseSession:
    @pytest.mark.asyncio
    async def test_close_session(self):
        sess = MagicMock()
        sess.proc.close = AsyncMock()
        await _close_session(sess)
        sess.proc.close.assert_awaited_once()


class TestStartSession:
    @pytest.mark.asyncio
    async def test_start_session_stores_workspace_runtime(self):
        workspace_runtime = MagicMock(name="workspace_runtime")
        ws = MagicMock(name="ws")
        proc = MagicMock(name="proc")
        runner = MagicMock()
        runner.start_program = AsyncMock(return_value=proc)
        inputs = ExecInput(skill="s", command="echo hi")

        session = await _start_session(
            runner=runner,
            tool_context=MagicMock(),
            inputs=inputs,
            ws=ws,
            workspace_runtime=workspace_runtime,
            rel_cwd="skills/s",
            env={"A": "1"},
        )

        assert isinstance(session, _ExecSession)
        assert session.workspace_runtime is workspace_runtime
        assert session.ws is ws
        assert session.proc is proc
        assert session.in_data is inputs


class TestCollectFinalResult:
    """Regression coverage for issue #350.

    ``_prepare_outputs`` requires ``workspace_runtime``.  A plain ``AsyncMock``
    swallows the missing-argument ``TypeError``, so the collection path silently
    degrades to empty files.  These tests bind a signature-strict stub so a
    dropped argument fails the assertions instead of being absorbed.
    """

    def _make_session(self, workspace_runtime, in_data=None, run_result=None):
        proc = MagicMock()
        proc.run_result = AsyncMock(return_value=run_result or MagicMock(
            stdout="done\n",
            stderr="",
            exit_code=0,
            timed_out=False,
            duration=0.0,
        ))
        return _ExecSession(
            proc=proc,
            ws=MagicMock(name="ws"),
            workspace_runtime=workspace_runtime,
            in_data=in_data or ExecInput(skill="s", command="echo hi", output_files=["out/result.txt"]),
        )

    def _make_run_tool(self, prepare_side_effect=None, files=None):
        run_tool = MagicMock()
        run_tool._prepare_outputs = AsyncMock(
            side_effect=prepare_side_effect,
            return_value=(files if files is not None else [], None),
        )
        run_tool._attach_artifacts_if_requested = AsyncMock()
        run_tool._merge_manifest_artifact_refs = MagicMock()
        return run_tool

    @pytest.mark.asyncio
    async def test_collect_final_result_passes_workspace_runtime_and_files(self):
        workspace_runtime = MagicMock(name="workspace_runtime")
        session = self._make_session(workspace_runtime)
        expected_files = [SkillRunFile(
            name="out/result.txt",
            content="ok",
            mime_type="text/plain",
            size_bytes=2,
        )]

        async def _prepare_outputs(ctx, ws, runtime, input_data):
            # Signature-strict: a missing argument raises TypeError here and
            # the production `except Exception` would degrade to empty files.
            assert runtime is workspace_runtime
            assert input_data.output_files == ["out/result.txt"]
            return list(expected_files), None

        run_tool = self._make_run_tool(prepare_side_effect=_prepare_outputs)
        ctx = MagicMock()
        result = await _collect_final_result(ctx, session, run_tool)

        assert result is not None
        assert result.exit_code == 0
        assert result.output_files == expected_files
        assert result.primary_output is not None
        assert result.primary_output.name == "out/result.txt"
        assert result.timed_out is False
        assert result.stderr == ""
        assert session.finalized is True
        assert session.final_result is result

        run_tool._prepare_outputs.assert_awaited_once()
        call_args = run_tool._prepare_outputs.await_args.args
        expected_input = SkillRunInput(
            skill="s",
            command="echo hi",
            output_files=["out/result.txt"],
        )
        assert call_args == (ctx, session.ws, workspace_runtime, expected_input)

    @pytest.mark.asyncio
    async def test_collect_final_result_returns_cached_when_finalized(self):
        session = self._make_session(MagicMock())
        cached = MagicMock(name="cached")
        session.finalized = True
        session.final_result = cached

        run_tool = self._make_run_tool()

        result = await _collect_final_result(MagicMock(), session, run_tool)
        assert result is cached
        run_tool._prepare_outputs.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_collect_final_result_omit_inline_content_clears_file_content(self):
        in_data = ExecInput(
            skill="s",
            command="echo hi",
            output_files=["out/result.txt"],
            omit_inline_content=True,
        )
        session = self._make_session(MagicMock(), in_data=in_data)
        collected = [SkillRunFile(
            name="out/result.txt",
            content="secret payload",
            mime_type="text/plain",
            size_bytes=14,
        )]
        run_tool = self._make_run_tool(files=collected)

        result = await _collect_final_result(MagicMock(), session, run_tool)

        assert result is not None
        assert len(result.output_files) == 1
        assert result.output_files[0].content == ""
        assert result.output_files[0].name == "out/result.txt"
        assert result.output_files[0].size_bytes == 14
        assert result.primary_output is not None
        assert result.primary_output.content == ""

    @pytest.mark.asyncio
    async def test_collect_final_result_propagates_timed_out(self):
        run_result = MagicMock(stdout="", stderr="", exit_code=124, timed_out=True, duration=1.5)
        session = self._make_session(MagicMock(), run_result=run_result)
        collected = [SkillRunFile(name="out/partial.txt", content="", mime_type="text/plain", size_bytes=0)]
        run_tool = self._make_run_tool(files=collected)

        result = await _collect_final_result(MagicMock(), session, run_tool)

        assert result is not None
        assert result.timed_out is True
        assert result.exit_code == 124
        assert result.duration_ms == 1500
        # Failed + timed_out run drops empty output files (same as skill_run).
        assert result.output_files == []
        assert any("empty output_files" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_collect_final_result_fills_stderr(self):
        run_result = MagicMock(stdout="out\n", stderr="boom\n", exit_code=1, timed_out=False, duration=0.25)
        session = self._make_session(MagicMock(), run_result=run_result)
        run_tool = self._make_run_tool()

        result = await _collect_final_result(MagicMock(), session, run_tool)

        assert result is not None
        assert result.stdout == "out\n"
        assert result.stderr == "boom\n"
        assert result.duration_ms == 250

    @pytest.mark.asyncio
    async def test_collect_final_result_warns_on_collection_failure(self):
        async def _prepare_outputs(ctx, ws, runtime, input_data):
            raise RuntimeError("glob failed")

        session = self._make_session(MagicMock())
        run_tool = self._make_run_tool(prepare_side_effect=_prepare_outputs)

        result = await _collect_final_result(MagicMock(), session, run_tool)

        assert result is not None
        assert result.output_files == []
        assert any("output collection failed" in w for w in result.warnings)
        # Still finalized so later polls do not retry a broken collection.
        assert session.finalized is True

    @pytest.mark.asyncio
    async def test_collect_final_result_exception_branch_survives_log_failure(self):
        session = self._make_session(MagicMock())
        session.proc.run_result = AsyncMock(side_effect=RuntimeError("session gone"))
        session.proc.log = AsyncMock(side_effect=RuntimeError("log gone"))
        session.exit_code = None
        run_tool = self._make_run_tool()

        result = await _collect_final_result(MagicMock(), session, run_tool)

        assert result is not None
        assert result.stdout == ""
        # Unknown exit code is surfaced via warnings instead of silent success.
        assert result.exit_code == 0
        assert any("exit code unavailable" in w for w in result.warnings)
        assert session.finalized is True
