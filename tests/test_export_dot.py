from examples import export_dot


def test_export_dot_writes_graphviz_to_stdout(capsys):
    export_dot.main([])

    assert capsys.readouterr().out.startswith("digraph")
