"""Command-line entry points, exercised the way a user would."""

import json

import pytest

from physim.cli import main


def run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr().out


def test_list_shows_solvers_problems_and_schemes(capsys):
    code, out = run(["list"], capsys)
    assert code == 0
    for expected in ("radau_iia5", "transmission_line", "qam64"):
        assert expected in out


def test_solve_prints_an_accuracy_line(capsys):
    code, out = run(["solve", "--problem", "damped_oscillator",
                     "--solver", "rk45", "--rtol", "1e-8"], capsys)
    assert code == 0
    assert "max relative error" in out.lower()


def test_solve_reports_failure_with_a_nonzero_exit_code(capsys):
    code, out = run(["solve", "--problem", "prothero_robinson",
                     "--solver", "forward_euler", "--steps", "100"], capsys)
    assert code != 0


def test_converge_reports_measured_order(capsys):
    code, out = run(["converge", "--problem", "exponential_decay",
                     "--solvers", "rk4", "--counts", "50,100,200,400"], capsys)
    assert code == 0
    assert "4.0" in out


def test_ber_accepts_a_range_specification(capsys):
    code, out = run(["ber", "--scheme", "qpsk", "--ebn0", "0:6:3",
                     "--target-errors", "100", "--max-symbols", "100000"],
                    capsys)
    assert code == 0
    assert out.count("\n") > 3


def test_link_prints_a_budget(capsys):
    code, out = run(["link", "--distance", "500", "--scheme", "qam16"], capsys)
    assert code == 0
    assert "margin" in out.lower()


def test_validate_quick_writes_json(tmp_path, capsys):
    out_path = tmp_path / "validation.json"
    code, out = run(["validate", "--quick", "--out", str(out_path)], capsys)
    assert code == 0
    payload = json.loads(out_path.read_text())
    assert payload["summary"]["accuracy_pass_rate"] == 1.0


def test_unknown_subcommand_is_rejected(capsys):
    with pytest.raises(SystemExit):
        main(["teleport"])
