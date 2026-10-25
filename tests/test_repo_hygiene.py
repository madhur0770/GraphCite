import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b|-----BEGIN [A-Z ]*PRIVATE KEY-----")


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    def test_neo4j_image_is_pinned_and_password_is_required_not_hardcoded(self):
        neo4j = self.compose["services"]["neo4j"]
        self.assertNotIn("latest", neo4j["image"])
        self.assertIn("${NEO4J_PASSWORD:?", neo4j["environment"]["NEO4J_AUTH"])
        self.assertNotIn("password", neo4j["environment"]["NEO4J_AUTH"].replace("NEO4J_PASSWORD", ""))

    def test_published_ports_bind_to_localhost_only(self):
        for name, service in self.compose["services"].items():
            for port in service.get("ports", []):
                self.assertTrue(str(port).startswith("127.0.0.1:"), f"{name}: {port}")

    def test_app_waits_for_healthy_database(self):
        self.assertEqual(self.compose["services"]["app"]["depends_on"]["neo4j"]["condition"], "service_healthy")

    def test_ollama_is_optional(self):
        self.assertEqual(self.compose["services"]["ollama"]["profiles"], ["ollama"])


class HygieneTests(unittest.TestCase):
    def test_env_files_are_ignored_but_example_is_not(self):
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".env", lines)
        self.assertIn("!.env.example", lines)

    def test_example_env_has_no_real_secret(self):
        self.assertIn("change-me", (ROOT / ".env.example").read_text(encoding="utf-8"))

    def test_no_credential_like_strings_committed(self):
        skip = {".git", "__pycache__"}
        for path in ROOT.rglob("*"):
            if path.is_file() and not skip.intersection(path.relative_to(ROOT).parts) and path.suffix in {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".html", ".example"}:
                self.assertIsNone(SECRET.search(path.read_text(encoding="utf-8", errors="ignore")), str(path))

    def test_dockerfile_runs_as_non_root(self):
        self.assertIn("USER appuser", (ROOT / "Dockerfile").read_text(encoding="utf-8"))

    def test_core_package_imports_without_optional_services(self):
        import subprocess, sys
        code = ("import sys, graphcite.pipeline, graphcite.demo_data;"
                "sys.exit(1 if any(m in sys.modules for m in ('streamlit','neo4j','ollama')) else 0)")
        self.assertEqual(subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode, 0)


if __name__ == "__main__":
    unittest.main()
