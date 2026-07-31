from tools.prompts.reader import PromptEntry
from tools.prompts.render import render_catalog

def test_render_catalog():
    e = PromptEntry("core_extractors.yaml", "function_type", ["classification"], ["enrichment"],
                    ["declarative", "interrogative"], ["enrichment"])
    out = render_catalog([e])
    assert "## core_extractors.yaml" in out
    assert "function_type" in out and "probabilistic" in out
    assert "classification" in out and "enrichment" in out
    assert "declarative" in out
