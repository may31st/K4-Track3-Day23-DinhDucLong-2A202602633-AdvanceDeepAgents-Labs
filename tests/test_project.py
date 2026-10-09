"""Offline contract tests; no API credentials, containers or paid calls required."""
import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

import check_citations
import research
import tools
import agents
from finalize_citations import finalize


SOURCES = [{"n": 1, "id": "a", "url": "https://example.org/a", "title": "A", "date": "2025", "source": "web"}]
REPORT = "# Survey\nA claim [1].\n\n## References\n[1] A. https://example.org/a (2025)\n"


class CitationsTests(unittest.TestCase):
    def test_valid_report(self):
        self.assertEqual(check_citations.check(REPORT, SOURCES), [])

    def test_source_family_urls_are_checked_inside_sandbox_validator(self):
        for family, url in [('arxiv', 'https://arxiv.org/abs/2501.00001'),
                            ('hf-search', 'https://huggingface.co/papers/2501.00001'),
                            ('hf-daily', 'https://huggingface.co/papers/2501.00001')]:
            with self.subTest(family=family):
                report = REPORT.replace('https://example.org/a', url)
                self.assertEqual(check_citations.check(report, [dict(SOURCES[0], source=family, url=url)]), [])
                problems = check_citations.check(REPORT, [dict(SOURCES[0], source=family)])
                self.assertTrue(any('family' in p for p in problems))

    def test_malformed_source_family_returns_problem(self):
        for family in (['web'], {'name': 'web'}, None):
            with self.subTest(family=family):
                self.assertTrue(check_citations.check(REPORT, [dict(SOURCES[0], source=family)]))

    def test_invalid_sources_and_references_are_rejected(self):
        cases = [([], REPORT), ({}, REPORT), ([None], REPORT),
                 ([dict(SOURCES[0], n=True)], REPORT),
                 ([dict(SOURCES[0], url="file:///a")], REPORT),
                 (SOURCES * 2, REPORT), (SOURCES, REPORT.replace("A claim [1]", "A claim [2]")),
                 (SOURCES, REPORT.replace("A claim [1]", "A claim")),
                 (SOURCES, REPORT.replace("## References", "## Other")),
                 (SOURCES, REPORT + "[1] A. https://example.org/a\n"),
                 (SOURCES, REPORT.replace("[1] A.", "[2] A.")),
                 (SOURCES, REPORT.replace("[1] A. https://example.org/a", "[1] A. https://example.org/b")),
                 (SOURCES, REPORT + "[2] B. https://example.org/b\n"),
                 (SOURCES, REPORT.replace("[1] A.", "[1] https://example.org/b A."))]
        for sources, report in cases:
            with self.subTest(sources=sources, report=report):
                self.assertTrue(check_citations.check(report, sources))

    def test_grouped_citations_and_code_exclusion(self):
        sources = SOURCES + [dict(SOURCES[0], n=2, url="https://example.org/b")]
        references = "\n## References\n[1] A https://example.org/a\n[2] B https://example.org/b\n"
        for group in ("[1, 2]", "[1-2]", "[1–2]"):
            body = "Claim " + group + ". `ignore [99]`\n~~~python\n[88]\n~~~\n[77](https://example.org)"
            self.assertEqual(check_citations.check(body + references, sources), [])
        self.assertTrue(check_citations.check("`[1]`\n" + references, sources))

    def test_finalizer_output_passes_validator(self):
        body, sources, errors = finalize("Claim [4, 7].", [dict(SOURCES[0], n=4), dict(SOURCES[0], n=7)])
        self.assertEqual(errors, [])
        self.assertEqual(check_citations.check(body, sources), [])

    def test_finalizer_handles_parenthesized_and_punctuated_urls(self):
        for url in ('https://en.wikipedia.org/wiki/World_model_(AI)', 'https://example.org/path?a=1;'):
            body, sources, errors = finalize('Claim [1].', [dict(SOURCES[0], url=url)])
            self.assertEqual(errors, [])
            self.assertEqual(check_citations.check(body, sources), [])

    def test_malformed_citation_returns_problem_instead_of_crashing(self):
        self.assertTrue(check_citations.check(REPORT.replace('A claim [1]', 'A claim [1-2-3]'), SOURCES))


class RetryTests(unittest.TestCase):
    def test_retries_and_caps_server_delay_without_last_sleep(self):
        calls = []
        def fail():
            calls.append(1)
            raise tools.RetryableError("busy", retry_after=100)
        with patch("tools.time.sleep") as sleep:
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(fail, attempts=3, cap=4)
        self.assertEqual(len(calls), 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [4, 4])

    def test_exponential_delay_and_recovery(self):
        calls = []
        def recover():
            calls.append(1)
            if len(calls) < 4:
                raise tools.RetryableError("busy")
            return "recovered"
        with patch("tools.time.sleep") as sleep:
            self.assertEqual(tools.with_retry(recover, base=2, cap=5), "recovered")
        delays = [c.args[0] for c in sleep.call_args_list]
        self.assertTrue(2 <= delays[0] <= 3)
        self.assertTrue(4 <= delays[1] <= 5)
        self.assertEqual(delays[2], 5)

    def test_non_retryable_error_is_not_retried(self):
        with patch("tools.time.sleep") as sleep:
            with self.assertRaises(ValueError):
                tools.with_retry(lambda: int("invalid"))
        self.assertEqual(sleep.call_count, 0)


class ToolTests(unittest.TestCase):
    def response(self, *, data=None, text=None, status=200, headers=None):
        return httpx.Response(status, json=data if text is None else None, text=text,
                              headers=headers, request=httpx.Request("GET", "https://example.org"))

    def test_arxiv_normalizes_atom_and_empty_query(self):
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v2</id><published>2025-01-01T00:00:00Z</published><title>A\n Paper</title><summary>Some\n evidence</summary></entry></feed>'
        with patch("tools.httpx.get", return_value=self.response(text=xml)):
            records = json.loads(tools.arxiv_search.invoke({"query": 'world: "model" AND', "max_results": 3}))
        self.assertEqual(records[0]["url"], "https://arxiv.org/abs/2501.00001")
        self.assertEqual(records[0]["title"], "A Paper")
        self.assertEqual(tools.arxiv_search.invoke({"query": '"::"'}), "NO RESULTS")

    def test_huggingface_mapping_filter_and_sort(self):
        items = [{"paper": {"id": "a", "title": "World model", "summary": "long", "ai_summary": "short", "upvotes": 3}},
                 {"paper": {"id": "b", "title": "World", "summary": "model", "upvotes": 9}}, {"paper": {}}]
        with patch("tools.httpx.get", return_value=self.response(data=items)):
            daily = json.loads(tools.hf_daily_papers.invoke({"keyword": "world"}))
            search = json.loads(tools.hf_search_papers.invoke({"query": "world"}))
        self.assertEqual([r["id"] for r in daily], ["b", "a"])
        self.assertEqual(search[0]["summary"], "short")
        self.assertEqual(search[0]["url"], "https://huggingface.co/papers/a")

    def test_all_tools_return_error_on_network_failure(self):
        with patch("tools.httpx.get", side_effect=ValueError("bad response")), patch("tools.httpx.post", side_effect=ValueError("bad response")):
            for tool, args in [(tools.arxiv_search, {"query": "world"}), (tools.hf_daily_papers, {}),
                               (tools.hf_search_papers, {"query": "world"}), (tools.web_search, {"query": "world"}),
                               (tools.web_fetch, {"url": "https://example.org"})]:
                self.assertTrue(tool.invoke(args).startswith("ERROR:"))

    def test_exa_sse_redaction_and_rate_limit_retry(self):
        limited = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "Rate limit exceeded"}], "_meta": {"rateLimited": True}}}
        success = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "Evidence https://example.org"}]}}
        replies = [self.response(text="event: message\ndata: " + json.dumps(r) + "\n\n") for r in (limited, success)]
        with patch("tools.httpx.post", side_effect=replies), patch("tools.time.sleep"):
            self.assertEqual(tools.web_search.invoke({"query": "world"}), "Evidence https://example.org")
        with patch.dict("os.environ", {"EXA_API_KEY": "private-test-value"}), patch("tools.httpx.post", side_effect=ValueError("private-test-value")):
            self.assertNotIn("private-test-value", tools.web_fetch.invoke({"url": "https://example.org"}))

    def test_exa_does_not_retry_legitimate_rate_limit_research_content(self):
        for metadata in ({'rateLimitExceeded': False}, {'rate_limited': False}):
            response = self.response(data={'jsonrpc': '2.0', 'id': 1, 'result': {
                'content': [{'type': 'text', 'text': 'This paper studies rate limits in API design.'}],
                '_meta': metadata}})
            with self.subTest(metadata=metadata), patch('tools.httpx.post', return_value=response), patch('tools.time.sleep'):
                self.assertEqual(tools.web_search.invoke({'query': 'rate limits'}),
                                 'This paper studies rate limits in API design.')

    def test_http_retry_after_and_permanent_failure(self):
        replies = [self.response(status=429, headers={'Retry-After': '7'}), self.response(data=[])]
        with patch('tools.httpx.get', side_effect=replies), patch('tools.time.sleep') as sleep:
            self.assertEqual(tools.hf_search_papers.invoke({'query': 'world'}), 'NO RESULTS')
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [7])
        with patch('tools.httpx.get', return_value=self.response(status=401)), patch('tools.time.sleep') as sleep:
            self.assertTrue(tools.hf_daily_papers.invoke({}).startswith('ERROR:'))
        self.assertEqual(sleep.call_count, 0)


class ResearchTests(unittest.TestCase):
    def test_openai_clients_have_bounded_timeout_and_retries(self):
        # A real constructor makes this check cover the SDK client configuration.
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model='test-model', api_key='not-a-real-key')
        research.bound_model_requests(model)
        self.assertEqual(model.root_client.timeout, 120)
        self.assertEqual(model.root_async_client.timeout, 120)
        self.assertEqual(model.root_client.max_retries, 1)
        self.assertEqual(model.root_async_client.max_retries, 1)
        self.assertIs(model.client, model.root_client.chat.completions)
        self.assertIs(model.async_client, model.root_async_client.chat.completions)

    def test_progress_never_prints_inputs_or_model_configuration(self):
        output = io.StringIO()
        with redirect_stdout(output):
            progress = research.RunProgress()
            progress.on_chat_model_start({'api_key': 'private-value'}, [['private prompt']])
            progress.on_tool_start({'name': 'web_search', 'api_key': 'private-value'}, 'private input')
        self.assertIn('web_search', output.getvalue())
        self.assertNotIn('private', output.getvalue())

    def test_missing_report_error_includes_lead_failure_explanation(self):
        from contextlib import contextmanager
        @contextmanager
        def sandbox():
            def execute(command):
                return SimpleNamespace(exit_code=1 if 'finalize_citations.py' in command and command.startswith('python3') else 0,
                                       output='report missing')
            yield SimpleNamespace(execute=execute)
        messages = [{'tool_calls': [{'name': 'task'}] * 3},
                    {'content': 'Unable to get a third source family.'}]
        agent = SimpleNamespace(invoke=lambda *args, **kwargs: {'messages': messages})
        errors = io.StringIO()
        with patch.dict('os.environ', {'LAB_MODEL': 'test-model'}), patch('research.make_model', return_value=object()), patch('research.open_sandbox', sandbox), patch('research.upload'), patch('research.build_lead_agent', return_value=agent), redirect_stderr(errors), redirect_stdout(io.StringIO()):
            self.assertEqual(research.main('topic'), 1)
        self.assertIn('Unable to get a third source family', errors.getvalue())

    def test_prompt_carries_topic_as_data(self):
        self.assertIn('world model', research.build_prompt('world model'))

    def test_failure_cleans_up_sandbox_and_saves_nothing(self):
        from contextlib import contextmanager
        lifecycle = []
        @contextmanager
        def sandbox():
            lifecycle.append('opened')
            try:
                yield SimpleNamespace(execute=lambda command: SimpleNamespace(exit_code=0, output=''))
            finally:
                lifecycle.append('closed')
        with patch.dict('os.environ', {'LAB_MODEL': 'test-model'}), patch('research.make_model', return_value=object()), patch('research.open_sandbox', sandbox), patch('research.upload'), patch('research.build_lead_agent', side_effect=RuntimeError('test failure')):
            self.assertEqual(research.main('topic'), 1)
        self.assertEqual(lifecycle, ['opened', 'closed'])

    def test_safe_slug(self):
        for topic, expected in [("../../x", "x"), ("", "topic"), ("Survey ABOUT world model", "survey-about-world-model")]:
            self.assertEqual(research.slugify(topic), expected)
        self.assertLessEqual(len(research.slugify("a" * 100)), 60)

    def test_message_usage_summary(self):
        messages = [SimpleNamespace(tool_calls=[{"name": "task"}, {"name": "execute"}], usage_metadata={"input_tokens": 11, "output_tokens": 5}),
                    {"tool_calls": [{"name": "task"}], "usage_metadata": {"input_tokens": 3, "output_tokens": 2}}]
        summary = research.summarize(messages, 1.23, "test-model")
        self.assertEqual(summary["subagent_calls"], 2)
        self.assertEqual(summary["tool_calls"], {"task": 2, "execute": 1})
        self.assertEqual(summary["tokens"], {"input": 14, "output": 7})

    def test_saves_exact_downloaded_bytes(self):
        files = {research.REPORT_PATH: REPORT.encode(), research.SOURCES_PATH: json.dumps(SOURCES).encode()}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder, patch("research.download", return_value=files):
            output = research.save_outputs(None, "Topic", [], 1, "test", Path(folder))
            self.assertEqual(output.read_bytes(), files[research.REPORT_PATH])
            self.assertEqual(output.with_suffix(".sources.json").read_bytes(), files[research.SOURCES_PATH])
            self.assertEqual(json.loads(output.with_suffix(".meta.json").read_text())["source_families"], ["web"])

    def test_invalid_download_writes_nothing(self):
        for report, sources in [(None, b"[]"), (b" ", b"[]"), (b"report", b"{broken"), (REPORT.encode(), b"{}"), (REPORT.encode(), b"[]")]:
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder, patch("research.download", return_value={research.REPORT_PATH: report, research.SOURCES_PATH: sources}):
                with self.assertRaises(RuntimeError):
                    research.save_outputs(None, "topic", [], 1, "test", Path(folder))
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_empty_topic_exits_with_usage(self):
        self.assertEqual(research.main(" "), 2)


class AgentTests(unittest.TestCase):
    def test_compiled_agent_can_plan_without_api_calls(self):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage
        class ToolModel(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self
        model = ToolModel(messages=iter([
            AIMessage(content='', tool_calls=[{'name': 'write_todos', 'args': {
                'todos': [{'content': 'Verify workflow', 'status': 'completed'}]}, 'id': 'plan'}]),
            AIMessage(content='Finished planning'),
        ]))
        result = agents.build_lead_agent(None, model).invoke(
            {'messages': [{'role': 'user', 'content': 'Plan the workflow'}]}, config={'recursion_limit': 20})
        self.assertEqual(result['todos'], [{'content': 'Verify workflow', 'status': 'completed'}])
        self.assertEqual(result['messages'][-1].content, 'Finished planning')

    def test_subagents_expose_only_assigned_source_tools_with_limits(self):
        specs = agents.build_subagents()
        by_name = {s['name']: s for s in specs}
        self.assertEqual({t.name for t in by_name['researcher']['tools']},
                         {'arxiv_search', 'hf_daily_papers', 'hf_search_papers', 'web_search', 'web_fetch'})
        self.assertEqual([t.name for t in by_name['citation-checker']['tools']], ['web_fetch'])
        for spec in specs:
            self.assertTrue(spec['middleware'])
            self.assertTrue(all(m.run_limit > 0 for m in spec['middleware']))


if __name__ == "__main__":
    unittest.main()
