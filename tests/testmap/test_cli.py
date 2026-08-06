from tools.testmap.__main__ import main


def test_check_returns_zero(capsys):
    assert main(["check"]) == 0
    assert "testmap-check:" in capsys.readouterr().out


def test_verification_runs(capsys):
    assert main(["verification"]) == 0
