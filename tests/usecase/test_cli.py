from tools.usecase.__main__ import main


def test_check_returns_zero_non_blocking(capsys):
    assert main(["check"]) == 0
    assert "usecase-check:" in capsys.readouterr().out


def test_coverage_command_runs(capsys):
    assert (
        main(["coverage"]) == 0
    )  # prints one line per use-case; may be empty pre-corpus
