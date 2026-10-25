**Title:** Exact pin `cloudpickle==2.2.1` conflicts with current joblib (`cloudpickle>=3.0`)

**Describe the bug**

amazon-braket-sdk 1.127.5 requires `cloudpickle==2.2.1`. joblib 1.6.0 (pulled in by scikit-learn) requires
`cloudpickle>=3.0`, so pip reports a conflict in any environment that has both the Braket SDK and a current
scikit-learn:

```
amazon-braket-sdk 1.127.5 requires cloudpickle==2.2.1, but you have cloudpickle 3.1.2 which is incompatible.
```

(or the reverse warning for joblib, depending on install order).

**To reproduce**

```bash
python -m venv env
env/bin/pip install amazon-braket-sdk scikit-learn
env/bin/pip check
```

**Expected behavior**

A version range (for example `cloudpickle>=2.2.1,<4`) if 3.x is compatible, so both can be installed without conflict.

**Additional context**

With cloudpickle 3.1.2 installed anyway, local simulation worked for us (`LocalSimulator`, `Circuit().ry(...)`, shots,
`measurement_counts`); we did not exercise the parts of the SDK that serialize functions (hybrid jobs), which is
presumably why the pin exists.

amazon-braket-sdk 1.127.5, joblib 1.6.0, scikit-learn 1.9.1, Python 3.12, Windows 11.
