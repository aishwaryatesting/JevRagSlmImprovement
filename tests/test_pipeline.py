from app.pipeline import run_pipeline

def test_health_pipeline():
    result = run_pipeline("Can I return opened headphones?")
    assert "checks" in result
    assert result["checks"]

def test_headphone_policy_supported():
    result = run_pipeline("What is the return policy for opened headphones?")
    assert result["checks"][0]["verdict"] == "SUPPORTED"
