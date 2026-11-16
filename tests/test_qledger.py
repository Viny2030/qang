from qang.qledger import parse_appendix_h, primary_status, STATUSES, function_names
T = "## 9.9 Apéndice H\n\n| id | familia | fórmula | función | estatus |\n| :--- | :--- | :--- | :--- | :--- |\n| `a` | x | y | `f` | `exact`; hw `unverified` |\n| `b` | x | y | `verify_cap09.py` §4 | `no gain` |\n\n## Fin\n"
def test_parse(): r = parse_appendix_h(T, 9); assert [x["id"] for x in r] == ["a", "b"]
def test_primary(): r = parse_appendix_h(T, 9); assert primary_status(r[0]["status"]) == "exact" and primary_status(r[1]["status"]) == "no gain"
def test_names(): assert function_names("`verify_cap09.py` §4") == ["verify_cap09.py"] and function_names("`AffineChannel.then`") == ["then"]
def test_status_list(): assert "measured gain" in STATUSES and len(STATUSES) == 8
